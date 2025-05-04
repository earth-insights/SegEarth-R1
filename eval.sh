export CUDA_VISIBLE_DEVICES=0
python SegEarthR1/eval_and_test/eval.py \
  --model_path your_model_path \
  --data_split "val" \
  --version "llava_phi" \
  --mask_config "SegEarthR1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "EarthReason" \
