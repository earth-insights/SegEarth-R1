# Evaluation🔍

Run `scripts/eval.sh` to evaluate the model. Make sure to update the script paths to match your setup.

```bash
export CUDA_VISIBLE_DEVICES=0
python segearth_r1/eval_and_test/eval.py \
  --model_path checkpoint/segearth_r1_EerthReason \
  --data_split "val" \
  --version "llava_phi" \
  --mask_config "segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "EarthReason" \
```
