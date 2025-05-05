# Training🚀

You can run `scripts/train.sh` to start training. Don't forget to modify the parameters in the script to match your own paths.

```bash
export DISABLE_ADDMM_CUDA_LT=1
deepspeed --master_port 29500 --include localhost:0 segearth_r1/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path "your_data_path" \
    --model_name_or_path "your_phi-1_5_path" \
    --version "llava_phi" \
    --vision_tower "your_swin_base_path" \
    --mask_config "segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoint/segearth_r1_EerthReason \
    --num_train_epochs 15 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'referring' \
    --freeze_mm_mlp_adapter False \
    --bf16 True \
    --train_backbone False \
    --mm_projector_type "compression_connector" \
    --dataset_type 'EarthReason' \
```
