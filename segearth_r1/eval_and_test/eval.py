import torch
from enum import Enum
from tqdm import tqdm
import numpy as np
from eval_dataset.RS_val_dataset import DataCollector, RRSISDDataset, ReasonSegDataset, RefSegRSDataset
from segearth_r1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX ,ANSWER_TOKEN_INDEX
from segearth_r1.model.builder import load_pretrained_model
from segearth_r1.utils import disable_torch_init
from segearth_r1.mm_utils import get_model_name_from_path
from segearth_r1 import conversation as conversation_lib
from torch.utils.data import DataLoader
from typing import Optional
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
    
def compute_metric(intersection_meter,union_meter,acc_iou_meter, pr_meters, cur_res, gt):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i, result in enumerate(cur_res):
        gt_mask = gt[i].squeeze(0).int().cuda().contiguous()
        pred_masks = result["pred_masks"].int().cuda().contiguous()
        if result["scores"]:
            scores = result["scores"]
            scores = scores.cpu().numpy()
            topk_scores,idx = torch.topk(torch.tensor(scores),1)
            idx = idx.cpu().numpy()
            topk_preds = pred_masks[idx,:]
        else:
            topk_preds = None
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        
        if topk_preds:
            for i, pred_ in enumerate(topk_preds):
                intersection, union, _ = intersectionAndUnionGPU(
                    pred_masks, gt_mask, 2, ignore_index=255
                )
            
                intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                acc_iou = intersection / (union + 1e-5)
                acc_iou[union == 0] = 1.0  # no-object target
                fore_acc_iou = acc_iou[1]
        else:
            intersection, union, _ = intersectionAndUnionGPU(
                pred_masks, gt_mask, 2, ignore_index=255
            )
            
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]   
            
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        
        for threshold in thresholds:
            if max_iou[1] > threshold:
                pr_meters[threshold].update(1.0, n=1)
            else:
                pr_meters[threshold].update(0.0, n=1)
        
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    base_data_path: str = "/root/siton-data-412581749c3f4cfea0d7c972b8742057/data"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./segearth_r1/mask_config/maskformer2_swin_large.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r1'
    version: str = 'llava_phi'
    segmentation: bool = True
    eval_batch_size: int = 5
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    data_split: Optional[str] = field(default="val")
    use_seg_query: bool = False
    dataset_type: Optional[str] = field(default="RRSIS-D")

def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = data_args.model_path
    model_name = get_model_name_from_path(model_path)
    print(f'current model is {model_path}')
    tokenizer, model, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, 
                                                          use_seg_query = data_args.use_seg_query, device='cuda')
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    if data_args.dataset_type == 'RRSIS-D':
        eval_dataset = RRSISDDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split = data_args.data_split,
        )
    if data_args.dataset_type == 'EarthReason':
        eval_dataset = ReasonSegDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split=data_args.data_split,
        )
    if data_args.dataset_type == 'RefSegRS':
        eval_dataset = RefSegRSDataset(
            base_data_path=data_args.base_data_path,
            tokenizer=tokenizer,
            split=data_args.data_split,
        )
    data_collator = DataCollector(
        tokenizer=tokenizer,
    )
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device,dtype=torch.float).eval()
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    pr_meters = {
        threshold: AverageMeter(f"Pr@{threshold}", ":6.3f", Summary.AVERAGE)
        for threshold in thresholds
    }
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            gt = inputs["masks"]
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            if 'token_answer_id' in inputs:
                inputs['token_answer_id'] = [ids.to(device) for ids in inputs['token_answer_id']]
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    masks=inputs['masks'],
                    token_refer_id = inputs['token_refer_id'],
                    refer_embedding_indices=inputs['refer_embedding_indices'],
                    labels=inputs['labels'],
                    token_answer_id=inputs['token_answer_id'],
                    answer_embedding_indices=inputs['answer_embedding_indices']
                    )
            else:
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    masks=inputs['masks'],
                    token_refer_id = inputs['token_refer_id'],
                    refer_embedding_indices=inputs['refer_embedding_indices'],
                    labels=inputs['labels'],
                    token_answer_id=None,
                    answer_embedding_indices=None
                    )
            compute_metric(intersection_meter,union_meter,acc_iou_meter, pr_meters, outputs, gt)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] 
    giou = acc_iou_meter.avg[1] 
    print(
            "ciou: {:.4f}, giou: {:.4f}".format(ciou, giou)
        )
    print(
            "IoU Thresholds: " + 
            ", ".join([f"@{t}: {m.avg:.4f}" for t, m in pr_meters.items()])
        )
    
if __name__ == "__main__":
    evaluation()