# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 多头注意力

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before # False

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # (256, 2048)
        self.dropout = nn.Dropout(dropout) # dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # (2048, 256)

        self.norm = nn.LayerNorm(d_model) # LayerNorm层

        self.activation = _get_activation_fn(activation) # 激活层
        self.normalize_before = normalize_before # False

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(
                                                                                                                           i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MultiScaleMaskedTransformerDecoderForOPTPreTrain(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            seg_norm=False,
            seg_concat=True,
            seg_proj=True,
            seg_fuse_score=False,
            use_seg_query=False,
    ):
        nn.Module.__init__(self)
        # positional encoding
        N_steps = hidden_dim // 2 # 256 / 2 =128
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads     # 8
        self.num_layers = dec_layers  # 9
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.use_seg_query = use_seg_query

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.seg_norm = seg_norm    # False
        self.seg_concat = seg_concat    # False
        self.seg_proj = seg_proj    # True
        self.seg_fuse_score = seg_fuse_score    # False
        if self.seg_norm:   # 改一下这里
            print('add seg norm for [SEG]')
            self.seg_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.class_name_proj_after_norm = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.SEG_norm = nn.LayerNorm(hidden_dim)
            self.class_name_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries  # 100
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)     # (100, 256)
        # self.query_feat = nn.Embedding(200, hidden_dim)
        # learnable query p.e. # 对于语义分割这边的query_embedding怎么办呢？
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)    # (100, 256)
        # self.query_embed = nn.Embedding(200, hidden_dim)
        
        self.SEG_query_embed = nn.Embedding(num_queries + 1, hidden_dim) # (101, 256)
        # self.SEG_query_embed = nn.Embedding(200 + 1, hidden_dim)
        
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # self.query_feat = nn.Embedding(200, hidden_dim)
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.query_embed = nn.Embedding(200, hidden_dim)
        
        self.new_query_embed = nn.Embedding(1, hidden_dim) # (1, 256)
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim) # (3, 256)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:  # 无
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential()) # input_proj 啥也没有

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)  # 三层nn.Linear
        self.SEG_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)  # 两层nn.Linear
        self.CLASS_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 2) # 两层nn.Linear
        # self.class_embed = nn.Linear(hidden_dim, 1)
        # self.class_embed_sim = nn.Linear(hidden_dim, 81)

    def forward(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None, class_name_embedding=None):
        if self.seg_concat:
            return self.forward_concat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding)
        else:
            return self.forward_woconcat(x, mask_features, mask, seg_query, SEG_embedding, class_name_embedding)

    def forward_concat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
                       class_name_embedding=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.SEG_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        if seg_query is None:
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            output = seg_query.permute(1, 0, 2)

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[0],
                                                                                             SEG_embedding=SEG_embedding,
                                                                                             class_name_embedding=class_name_embedding,
                                                                                             )
        predictions_SEG_class.append(SEG_class)
        predictions_class_name_class.append(class_name_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            output = torch.cat([SEG_embedding.transpose(0, 1), output], 0)
            SEG_mask = torch.zeros((attn_mask.shape[0], 1, attn_mask.shape[-1]), dtype=torch.bool,
                                   device=attn_mask.device)
            attn_mask = torch.cat([SEG_mask, attn_mask], dim=1)
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output[1:]
            SEG_embedding = output[0].unsqueeze(0).transpose(0, 1)
            SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=
                size_list[(
                                  i + 1) % self.num_feature_levels],
                SEG_embedding=SEG_embedding,
                class_name_embedding=class_name_embedding
                )
            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)
            
        assert len(predictions_SEG_class) == self.num_layers + 1

        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask
            )
        }
        return out

    def forward_woconcat(self, x, mask_features, mask=None, seg_query=None, SEG_embedding=None,
                         class_name_embedding=None): # x:[[batch_size, 256, 32, 32], [batch_size, 256, 64, 64], [batch_size, 256, 128, 128]] 
        # x is a list of multi-scale feature                                     # mask_features: [batch_size, 256, 256, 256]
        assert len(x) == self.num_feature_levels                                 # seg_query: [batch_size, 100, mask_dim(256)], SEG_embedding: [batch_size, 1, mask_dim(256)]
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask # None
        if class_name_embedding is not None:
            SEG_embedding = class_name_embedding

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(x[i].dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        # size_list: [[32, 32], [64, 64], [128, 128]], pos: [[32 * 32, batch_size, 2], [64 * 64, batch_size, 2], [128 * 128, batch_size, 2]], src: [[32 * 32, batch_size, 2], [64 * 64, batch_size, 2], [128 * 128, batch_size, 2]]
        _, bs, _ = src[0].shape # batch_size

        # QxNxC
        if self.use_seg_query:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            query_embed = torch.zeros(
                self.new_query_embed.weight.shape[0], bs, self.new_query_embed.weight.shape[-1], 
                device=SEG_embedding.device, dtype=SEG_embedding.dtype
                )
            # query_embed = self.new_query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # query_embed: [1, batch_size, mask_dim(256)]
        if seg_query is None:
            # output = self.new_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            output = SEG_embedding.permute(1, 0, 2)
        else:
            output = seg_query.permute(1, 0, 2) # output: [100, batch_size, mask_dim(256)]

        predictions_SEG_class = []
        predictions_class_name_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        if self.use_seg_query:
            SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                            mask_features,
                                                                                            attn_mask_target_size=
                                                                                            size_list[
                                                                                                        0],
                                                                                            SEG_embedding=SEG_embedding,
                                                                                            class_name_embedding=None,
                                                                                            )
        else:
            SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                            mask_features,
                                                                                            attn_mask_target_size=
                                                                                            size_list[
                                                                                                        0],
                                                                                            SEG_embedding=None,
                                                                                            class_name_embedding=None,
                                                                                            )
        predictions_SEG_class.append(SEG_class) # SEG_class: [batch_size, 100, 1] -> None
        predictions_class_name_class.append(class_name_class) # class_name_class: None
        predictions_mask.append(outputs_mask) # outputs_mask: [batch_size, 100, 256, 256] -> [batch_size, 1, 256, 256]

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            if self.use_seg_query:
                SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=
                    size_list[(
                                      i + 1) % self.num_feature_levels],
                    SEG_embedding=SEG_embedding,
                    class_name_embedding=None
                    )
            else:
                SEG_class, class_name_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                    output, mask_features,
                    attn_mask_target_size=
                    size_list[(
                                      i + 1) % self.num_feature_levels],
                    SEG_embedding=None,
                    class_name_embedding=None
                    )
            predictions_SEG_class.append(SEG_class)
            predictions_class_name_class.append(class_name_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_SEG_class) == self.num_layers + 1
        # predictions_SEG_class: [batch_size, 100, 1], predictions_class_name_class: 对于参考分割来说是None, predictions_mask: [batch_size, 100, 256, 256], predictions_region_class: 对于语义分割和参考分割都是None
        out = {
            'pred_SEG_logits': predictions_SEG_class[-1],
            'pred_class_name_logits': predictions_class_name_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_SEG_class, predictions_class_name_class, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, SEG_embedding=None,
                                 class_name_embedding=None):
        decoder_output = self.decoder_norm(output) # output是seg_query [4, 100, 256]
        decoder_output = decoder_output.transpose(0, 1)

        if SEG_embedding is not None:
            if self.seg_proj:
                decoder_seg_output = self.SEG_proj(decoder_output)
            else:
                decoder_seg_output = decoder_output
            if self.seg_norm:
                SEG_embedding = self.SEG_norm(SEG_embedding)
                SEG_embedding = self.seg_proj_after_norm(SEG_embedding)
            SEG_class = torch.einsum('bld,bcd->blc', decoder_seg_output, SEG_embedding)
        else:
            SEG_class = None
        # SEG_class = F.cosine_similarity(decoder_seg_output, SEG_embedding, dim=-1, eps=1e-6).unsqueeze(-1)
        if class_name_embedding is not None:
            # decoder_class_output = decoder_output.detach()
            decoder_class_output = decoder_output
            if self.seg_proj:
                decoder_class_output = self.CLASS_proj(decoder_class_output)
            else:
                decoder_class_output = decoder_class_output
            if self.seg_norm:
                class_name_embedding = self.class_name_norm(class_name_embedding)
                class_name_embedding = self.class_name_proj_after_norm(class_name_embedding)
            dot_product = torch.einsum('bld,bcd->blc', decoder_class_output, class_name_embedding)

            if self.seg_fuse_score:
                class_SEG_class = SEG_class.expand_as(dot_product)
                reverse_bg_mask = torch.ones_like(class_SEG_class).to(dtype=class_SEG_class.dtype,device=class_SEG_class.device)
                reverse_bg_mask[:,:,-1] = -reverse_bg_mask[:,:,-1]
                class_name_class = dot_product * class_SEG_class * reverse_bg_mask
            else:
                class_name_class = dot_product
        else:
            class_name_class = None
        # mask_embed = self.mask_embed(SEG_embedding)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask.float(), size=attn_mask_target_size, mode="bilinear",
                                  align_corners=False).to(mask_embed.dtype)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return SEG_class, class_name_class, outputs_mask, attn_mask
        # return outputs_mask, attn_mask
    @torch.jit.unused
    def _set_aux_loss(self, outputs_SEG_class, outputs_class_name_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        #     return [
        #         {"pred_logits": a, "pred_masks": b}
        #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        #     ]
        # else:
        #     return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

        return [
            {"pred_SEG_logits": a, "pred_class_name_logits": b, "pred_masks": c}
            for a, b, c in zip(outputs_SEG_class[:-1], outputs_class_name_class[:-1], outputs_seg_masks[:-1])
        ]






