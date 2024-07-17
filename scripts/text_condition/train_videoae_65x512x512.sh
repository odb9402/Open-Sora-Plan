export HF_HOME="/moai/cache"
export TRANSFORMERS_CACHE="/moai/cache"

MOREH_VISIBLE_DEVICE=1 \
python opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "/moai/cache" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/moai/model/open_sora_plan_ckpts/vae" \
    --video_data "metadata.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 32 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=16 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="65x512x512_10node_bs2_lr2e-5_4img" \
    --allow_tf32 \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --enable_tracker \
    --use_img_from_vid \
    --dataloader_num_workers 0 
    #--pretrained t2v.pt \
    #--resume_from_checkpoint "latest" \
