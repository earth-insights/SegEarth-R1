from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import fvcore.nn.weight_init as weight_init
from transformers import AutoConfig, AutoModelForCausalLM

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from segearth_r1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import \
    MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..multimodal_projector.builder import build_vision_projector
from ..multimodal_encoder.swin_trans import build_swin_b

from segearth_r1.model.mask_decoder.mask_criterion.pretrain_criterion import segearth_r1_criterion, hungarian_matcher_PSALM
from transformers import PhiModel, PhiForCausalLM, PhiConfig

from segearth_r1.model.language_model.projector import D_Projector

class LlavaConfig(PhiConfig):
    model_type = "llava_phi"

@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_SEG_class: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None


class segearth_r1Model(LlavaMetaModel, PhiModel): 
    config_class = LlavaConfig

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(segearth_r1Model, self).__init__(config)
        self.config.output_attentions = True
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size 
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower = build_swin_b(None)
            self.mm_projector = build_vision_projector(config)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):  
        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower 
        with_norm = model_args.with_norm # True                    
        with_layernorm = model_args.with_layernorm # False
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter if hasattr(model_args,
                                                                                'pretrain_mm_mlp_adapter') else None 
        projector_outdim = self.projector_outdim # 2048

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args,'swin_type','base') 
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower = build_swin_b(vision_tower)
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        vision_tower.hidden_size = 256
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'conv')
        print(f'current mm_project_type is {self.config.mm_projector_type}, the output dim is {projector_outdim}')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.with_norm = with_norm
        self.config.with_layernorm = with_layernorm
        self.config.projector_outdim = projector_outdim

        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config)
        else:
            print('exist mm_projector, skip init')

        if pretrain_mm_mlp_adapter is not None: # 加载mm_mlp_adapter
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # import ipdb;ipdb.set_trace()
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
            print('load mm_projector pth successfully')

        
class segearth_r1(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None, use_seg_query=False):
        super(segearth_r1, self).__init__(config)

        self.model = segearth_r1Model(config, mask_decoder_cfg)
        self.use_seg_query = use_seg_query
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode
        if use_seg_query:
            self.seg_query_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        
        self.refer_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        additional_dim = 512
        local_fea_dim = [128 * 2**i for i in [0, 1, 2, 3]]
        self.text_projector = nn.Linear(self.config.hidden_size, additional_dim)
        self.origin_SEG_token_projector = nn.Linear(self.config.hidden_size, additional_dim)
        self.local_project = nn.Linear(local_fea_dim[-1], additional_dim)
        self.SEG_token_projector = nn.Linear(2 * additional_dim, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.d_layers = D_Projector(dim=additional_dim, depth=1, dim_head=64, heads=8, ff_mult=1)       
        
        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:   # 训练过程中默认是False
            print('Initialize mask modules...')
            self.config.mask_decode_train = True # Train mask decode 
        self.seg_query = nn.Parameter(
            torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.config.hidden_size])) # 返回一个[100, 2048]的全0可训练的向量, 这个好像没什么用啊
        self.num_queries = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES # NUM_OBJECT_QUERIES为100
        self.num_classes = self.mask_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES # NUM_CLASSES为80
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES # 100
        input_shape = self.output_shape() # 这是一个函数
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape) # 初始化pixel_decoder
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}  # 获取keyword的参数
            def change_w(weights, old_name, new_name): # 为参数改名
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            
            # TODO: for swin large
            # if predictor_weights['query_embed.weight'].shape[0] == 200:
            #     predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    # def mask_token_processor()

    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower()(images)
        features_dict = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION   # True
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT   # 0.1  

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT   #  2.0 
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT # 5.0
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT # 5.0

        matcher = hungarian_matcher_PSALM(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS, # 12544 -> 112 * 112
        )

        weight_dict = {"loss_SEG_class": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight} 
        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS   # 10
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()}) 
            weight_dict.update(aux_weight_dict) 
        losses = ["SEG_labels", "class_name_labels", "masks"]
        self.criterion = segearth_r1_criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO, # 3.0
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO, # 0.75  
            device=self.device
        )
        self.size_divisibility = 32      
        self.sem_seg_postprocess_before_inference =  True
    
    def SEG_instance_inference(self, SEG_cls, mask_pred):
        image_size = mask_pred.shape[-2:] 
        scores = F.sigmoid(SEG_cls) if SEG_cls else None
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False) if SEG_cls else None, None 
        if SEG_cls is not None:
            mask_pred = mask_pred[topk_indices]
        elif mask_pred.shape[0] == 1:
            mask_pred = mask_pred[0]
        pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6) if SEG_cls else None
        scores = mask_scores_per_image * scores_per_image if SEG_cls else None
        return {
            "pred_masks": pred_masks,
            "scores": scores,
        }
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features[-1])
        return image_features

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM  
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM 
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 
        nheads = cfg.MODEL.MASK_FORMER.NHEADS 
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD 
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1 
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM 
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM 
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        seg_concat = False 
        use_seg_query = self.use_seg_query
        print(f'current seg concat mode: {seg_concat}, seg_norm: {seg_norm}, seg_proj: {seg_proj}, seg_fuse_score: {seg_fuse_score}')
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
                                                                     hidden_dim,
                                                                     num_queries,
                                                                     nheads,
                                                                     dim_feedforward,
                                                                     dec_layers,
                                                                     pre_norm,
                                                                     mask_dim,
                                                                     enforce_input_project,
                                                                     seg_norm,
                                                                     seg_concat,
                                                                     seg_proj,
                                                                     seg_fuse_score,
                                                                     use_seg_query)
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES    # ["res2", "res3", "res4", "res5"]
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        } 
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0], # 128
            "res3": num_features[1], # 256
            "res4": num_features[2], # 512
            "res5": num_features[3], # 1024
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape # {"res2":{"channel": 128, "stride": 4} ... }

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):  # input_shape: {"res2": {"channel": 128, "stride": 4} ... }
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE    # COMMON_STRIDE: 4
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT   # DROPOUT: 0.0
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS     # NHEADS: 8  
        transformer_dim_feedforward = 1024  # transformer_dim_feedforward: 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # TRANSFORMER_ENC_LAYERS: 6
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM     # CONVS_DIM: 256
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM # MASK_DIM: 256 
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"] 

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    def prepare_targets(self, targets, images): # images: [batch_size, 3, 1024, 1024]   targets: [image1(instances), image2(instances)]
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks # gt_masks: [1, 1024, 1024] 0 or 1
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer
    def concat_image_seg_embeds(self, input_id, img_feature, label, seg_query, seg_query_mask,
                                    refer_embedding_indices=None, refer_embedding=None,
                                    answer_embedding_indices=None, answer_embedding=None, token_answer_id=None):
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0] 
        if seg_query is not None:
            seg_query_indices = torch.where(input_id == SEG_TOKEN_INDEX)[0] 
        assert len(image_token_indices) == 1, 'not supporting multi image index'
        if seg_query is not None:
            assert len(seg_query_indices) == 1, 'not supporting multi seg index'
        cur_new_input_embeds = []
        if seg_query is not None:
            cur_new_seg_query_mask = []
        else: 
            cur_new_seg_query_mask = None
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        cur_refer_embedding_indices = [] if refer_embedding_indices is not None else None 
        chunks = [] 
        current_chunk = [] 

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

        cls_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX: 
                cur_new_input_embeds.append(img_feature) # [256, 2048]
                if seg_query is not None:
                    cur_new_seg_query_mask.append(torch.zeros(img_feature.shape[0])) # [256,] all zeros
                if refer_embedding_indices is not None: 
                    cur_refer_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype)) 
                if label is not None: 
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    ) 
            elif chunk_len == 1 and chunk[0] == SEG_TOKEN_INDEX:
                if seg_query is not None:
                    cur_new_input_embeds.append(seg_query) # seg_query: [100, 2048] all zeros
                    cur_new_seg_query_mask.append(torch.ones(seg_query.shape[0])) # [100,] all ones
                    if refer_embedding_indices is not None: 
                        cur_refer_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=label.device,
                                                                       dtype=label.dtype))
                    if label is not None: 
                        cur_new_label.append(
                            torch.full((seg_query.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype))
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX: 
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1: 
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed) # refer_embed: [refer_len, 2048]
                if seg_query is not None:
                    cur_new_seg_query_mask.append(torch.zeros(refer_embed.shape[0])) 
                if refer_embedding_indices is not None: 
                    cur_refer_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 1, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None: 
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            elif chunk_len == 1 and chunk[0] == ANSWER_TOKEN_INDEX: 
                answer_embed = answer_embedding
                if len(answer_embed.shape) == 1:
                    answer_embed = answer_embed.unsqueeze(0)
                cur_new_input_embeds.append(answer_embed)
                if seg_query is not None:
                    cur_new_seg_query_mask.append(torch.zeros(answer_embed.shape[0]))
                if refer_embedding_indices is not None: 
                    cur_refer_embedding_indices.append(
                        torch.full((answer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(token_answer_id)
                     
            else: 
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len])) # 
                if seg_query is not None:
                    cur_new_seg_query_mask.append(seg_query_mask[:chunk_len]) #
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(refer_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])

            input_id = input_id[chunk_len:] 
            if seg_query_mask is not None:
                seg_query_mask = seg_query_mask[chunk_len:] 
            if refer_embedding_indices is not None:
                refer_embedding_indices = refer_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds] 
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) 
        if label is not None: 
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        if seg_query is not None:
            cur_new_seg_query_mask = [x.to(device=self.device) for x in cur_new_seg_query_mask]
            cur_new_seg_query_mask = torch.cat(cur_new_seg_query_mask, dim=0)
        if refer_embedding_indices is not None: 
            cur_refer_embedding_indices = [x.to(device=self.device) for x in cur_refer_embedding_indices]
            cur_refer_embedding_indices = torch.cat(cur_refer_embedding_indices, dim=0)
        return cur_new_input_embeds, cur_new_label, cur_new_seg_query_mask, cur_refer_embedding_indices
    
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images,
            token_refer_id=None, token_answer_id=None, refer_embedding_indices=None, answer_embedding_indices=None, use_seg_query=False
    ): 
        vision_tower = self.get_vision_tower()
        seg_query_mask = torch.zeros_like(input_ids) if use_seg_query else None
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, seg_query_mask

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images) # image_features: [batch_size, 256, 2048]
        if use_seg_query:
            expanded_seg_query = self.seg_query.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = [] if seg_query_mask is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        # new_answer_embedding_indices = [] if answer_embedding_indices is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if use_seg_query:
                cur_seg_query_mask = seg_query_mask[batch_idx]
                cur_seg_query = expanded_seg_query[batch_idx]
            cur_image_feature = image_features[batch_idx]
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_answer_embedding_indices = answer_embedding_indices[batch_idx] if answer_embedding_indices is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: 
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None
            if token_answer_id is not None:
                cur_token_answer_id = token_answer_id[batch_idx]
            else:
                cur_token_answer_id = None


            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)
            cur_answer_embedding = self.embed_refer_ids(cur_token_answer_id)
            if use_seg_query:
                cur_input_embeds, cur_label, cur_seg_query_mask, cur_refer_embedding_indices = self.concat_image_seg_embeds(
                    input_id=cur_input_ids, # [seq_len, ]
                    img_feature=cur_image_feature, # [256, 2048]
                    label=cur_label, # [seq_len, ]
                    seg_query=cur_seg_query, # [100, 2048] all zeros
                    seg_query_mask=cur_seg_query_mask, # [seq_len, ] all zeros
                    refer_embedding_indices=cur_refer_embedding_indices, # [seq_len, ]
                    refer_embedding=cur_refer_embedding, # [refer_len, 2048]
                    answer_embedding_indices=cur_answer_embedding_indices,
                    answer_embedding=cur_answer_embedding,
                    token_answer_id = cur_token_answer_id,
                )
                assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]
            else:
                cur_input_embeds, cur_label, cur_seg_query_mask, cur_refer_embedding_indices = self.concat_image_seg_embeds(
                    input_id=cur_input_ids, # [seq_len, ]
                    img_feature=cur_image_feature, # [64, 2048]
                    label=cur_label, # [seq_len, ]
                    seg_query=None, # None
                    seg_query_mask=None, # None
                    refer_embedding_indices=cur_refer_embedding_indices, # [seq_len, ] 
                    refer_embedding=cur_refer_embedding,# [refer_len, 2048]  
                    answer_embedding_indices=cur_answer_embedding_indices,
                    answer_embedding=cur_answer_embedding,
                    token_answer_id = cur_token_answer_id,
                )
                
            new_input_embeds.append(cur_input_embeds) 
            if labels is not None: 
                new_labels.append(cur_label)
            if cur_seg_query_mask is not None:
                new_seg_query_masks.append(cur_seg_query_mask) 
            if refer_embedding_indices is not None: 
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds): 
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds: 
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0) 

            if labels is not None: 
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0) 
            
            if use_seg_query:
                new_seg_query_masks_align = []
                for new_seg_query_mask in new_seg_query_masks:
                    new_seg_query_mask = torch.cat(
                        (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                        dim=0)
                    new_seg_query_masks_align.append(new_seg_query_mask)
                new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)
            
            if refer_embedding_indices is not None: 
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0) 

            if attention_mask is not None: 
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0) 
                assert attention_mask.shape == new_labels.shape 

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0) if use_seg_query else None
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_refer_embedding_indices
    
    def get_SEG_embedding(self,hidden_states, refer_embedding_indices, return_all=False):
        refer_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, refer_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            current_pool_refer_state = self.refer_pooling(current_refer_state.transpose(-2, -1)).transpose(-2, -1)
            if return_all:
                current_pool_refer_state = torch.cat([current_pool_refer_state, current_refer_state], dim=0)

            refer_embedding_list.append(current_pool_refer_state)
        
        return torch.stack(refer_embedding_list, dim=0) if not return_all else refer_embedding_list
    
    def PyramidPoolAgg(self, image_features):
        B, C, H, W = image_features["res5"].shape
        return torch.cat([nn.functional.adaptive_avg_pool2d(v, (H, W)) for k, v in image_features.items()], dim=1)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True, 
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        masks: Optional[torch.FloatTensor] = None,
        token_refer_id: Optional[int] = None,
        token_answer_id: Optional[int] = None,
        refer_embedding_indices: Optional[List[int]] = None,
        answer_embedding_indices: Optional[List[int]] = None,
        dataset_type: Optional[str] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
        else:
            batch_dataset_type = []

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if batch_dataset_type != "mm_conv":
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images,
                token_refer_id, token_answer_id, refer_embedding_indices, answer_embedding_indices, self.use_seg_query)
        else:
            seg_query_mask = None
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ) 
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if batch_dataset_type == 'mm_conv' or batch_dataset_type == 'reason_seg': 
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)

        if batch_dataset_type == 'refer_seg' or batch_dataset_type == 'reason_seg':
            if self.use_seg_query:
                seg_query = self.seg_query_projector(self.get_seg_query(hidden_states, seg_query_mask))
            else:
                seg_query = None
            image_features = self.get_vision_tower_feature(images)
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
            if refer_embedding_indices is not None:
                SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices, return_all = True)
                origin_SEG_embedding = torch.cat([self.origin_SEG_token_projector(kk.unsqueeze(0)[:, 0:1]) for kk in SEG_embedding])
                local_vision = image_features["res5"].flatten(2).permute(0, 2, 1)
                local_vision = self.local_project(local_vision)   
                new_SEG_embedding = []
                for batch_idx, cur_SEG_embedding in enumerate(SEG_embedding):
                    cur_SEG_embedding = self.text_projector(cur_SEG_embedding.unsqueeze(0))
                    cur_SEG_embedding = self.d_layers(latents=cur_SEG_embedding.unsqueeze(1), 
                        x=local_vision[batch_idx:batch_idx+1].unsqueeze(1))
                    new_SEG_embedding.append(cur_SEG_embedding)
                new_SEG_embedding = torch.cat(new_SEG_embedding, dim=0)
                SEG_embedding = torch.cat((origin_SEG_embedding, new_SEG_embedding), dim=-1)
                SEG_embedding = self.SEG_token_projector(SEG_embedding) 
            else:
                SEG_embedding = None
                
            mask_outputs = self.predictor(
                    multi_scale_features, 
                    mask_features, 
                    None, 
                    seg_query,
                    SEG_embedding, 
                    None)

            if masks is not None: 
                targets = []
                for mask in masks:
                    target = {"labels": torch.tensor([1]), "masks": mask}
                    targets.append(target)             
                
                mask_losses = self.criterion(mask_outputs, targets)
                weight_dict = self.weight_dict

                loss_mask = 0.0
                loss_dice = 0.0
                loss_SEG_class = 0.0
                for k in list(mask_losses.keys()):
                    if k in weight_dict:
                        if mask_losses[k] is not None:
                            mask_losses[k] *= weight_dict[k]
                        if '_SEG' in k and mask_losses[k] is not None:
                            loss_SEG_class += mask_losses[k]
                        elif '_mask' in k:
                            loss_mask += mask_losses[k]
                        elif '_dice' in k:
                            loss_dice += mask_losses[k]
                    else:
                        mask_losses.pop(k)
                mask_loss = loss_mask + loss_dice + loss_SEG_class
                if isinstance(loss_SEG_class, float):
                    loss_SEG_class = torch.tensor(loss_SEG_class, device=mask_loss.device)
                if batch_dataset_type == 'refer_seg':
                    llm_loss = torch.tensor(0.0, device=mask_loss.device)
            loss = llm_loss + mask_loss
                
        if batch_dataset_type == 'mm_conv':
            loss_mask = torch.tensor(0.0, device=llm_loss.device)
            loss_dice = torch.tensor(0.0, device=llm_loss.device)
            loss_SEG_class = torch.tensor(0.0, device=llm_loss.device)
            loss = llm_loss       
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                loss_mask=loss_mask.detach(),
                loss_dice=loss_dice.detach(),
                loss_SEG_class=loss_SEG_class.detach(),
                loss_llm=llm_loss.detach(),
            )
        
        if batch_dataset_type == 'refer_seg' or batch_dataset_type == 'reason_seg':
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                loss_mask=loss_mask.detach(),
                loss_dice=loss_dice.detach(),
                loss_SEG_class=loss_SEG_class.detach(),
                loss_llm=llm_loss.detach(),
            )
        
    def mm_conv_prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # concat text and image embedding. prepare labels, IGNORE_INDEX for image tokens
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Align embedddings, labels, attn_mask from different sample into a batch
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def get_seg_query(self, hidden_states, seg_query_masks):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]

            for value in unique_query_value:
                current_query_mask = (sample_query_mask == value)
                current_query = sample_hidden_state[current_query_mask]

                seg_query_list.append(current_query)

        seg_query = torch.stack(seg_query_list, dim=0)

        return seg_query
    
    def eval_seg(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        masks: Optional[torch.Tensor] = None,
        token_refer_id: Optional[torch.LongTensor] = None,
        token_answer_id: Optional[torch.LongTensor] = None,
        refer_embedding_indices: Optional[torch.LongTensor] = None,
        answer_embedding_indices: Optional[torch.LongTensor] = None,
        is_thing_list: Optional[torch.Tensor] = None,
    ):  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images,
            token_refer_id, token_answer_id, refer_embedding_indices, answer_embedding_indices, self.use_seg_query)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        if self.use_seg_query:
            seg_query = self.seg_query_projector(self.get_seg_query(hidden_states, seg_query_mask))
        else:
            seg_query = None
        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:  
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices, return_all = True)
            origin_SEG_embedding = torch.cat([self.origin_SEG_token_projector(kk.unsqueeze(0)[:, 0:1]) for kk in SEG_embedding])
            local_vision = image_features["res5"].flatten(2).permute(0, 2, 1)
            local_vision = self.local_project(local_vision)
            new_SEG_embedding = []
            for batch_idx, cur_SEG_embedding in enumerate(SEG_embedding):
                cur_SEG_embedding = self.text_projector(cur_SEG_embedding.unsqueeze(0))
                cur_SEG_embedding = self.d_layers(latents=cur_SEG_embedding.unsqueeze(1), 
                    x=local_vision[batch_idx:batch_idx+1].unsqueeze(1))
                new_SEG_embedding.append(cur_SEG_embedding)
            new_SEG_embedding = torch.cat(new_SEG_embedding, dim=0)
            SEG_embedding = torch.cat((origin_SEG_embedding, new_SEG_embedding), dim=-1)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)  
        else:
            SEG_embedding = None
            
        mask_outputs = self.predictor(
                multi_scale_features, 
                mask_features, 
                None, 
                seg_query,
                SEG_embedding, 
                None)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None] * mask_pred_results.shape[0]
        for SEG_cls_result, mask_pred_result in zip(SEG_cls_results, mask_pred_results): 
            if SEG_cls_result is not None:
                SEG_cls_result = SEG_cls_result.to(mask_pred_result)
            if SEG_cls_result is None:
                results = self.SEG_instance_inference(None, mask_pred_result.float())
            else:
                results = self.SEG_instance_inference(SEG_cls_result.float(), mask_pred_result.float())
            processed_results.append(results)
            
        return processed_results

AutoConfig.register("llava_phi", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, segearth_r1)
