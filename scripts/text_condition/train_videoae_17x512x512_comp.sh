export WANDB_KEY=""
export ENTITY=""
export PROJECT="t2v-f17-512-img4-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5"
# accelerate launch \
    # --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    python opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.0.0/vae \
    --data_path /root/ddd/workspace/datasets/sharegpt4v_path_cap_64x512x512-vid16.json \
    --video_folder /root/ddd/workspace/datasets/vid16/videos \
    --sample_rate 1 \
    --num_frames 17 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=20 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --output_dir="t2v-f17-512-img4-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5" \
    --allow_tf32 \
    --pretrained LanguageBind/Open-Sora-Plan-v1.0.0/t2v.pt \
    --model_max_length 300 \
    --use_image_num 4 \
    --use_img_from_vid \
    # --use_deepspeed \
