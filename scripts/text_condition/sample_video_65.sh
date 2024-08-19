export PYTHONPATH=$PYTHONPATH:~/Open-Sora-Plan
MOREH_VISIBLE_DEVICE=2 \
python opensora/sample/sample_t2v.py \
    --model_path "/moai/cache/models--LanguageBind--Open-Sora-Plan-v1.1.0/snapshots/4d6a61ab745d7499236d8650b21676aae71271a4/65x512x512/diffusion_pytorch_model.pt" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --version 65x512x512 \
    --num_frames 65 \
    --height 512 \
    --width 512 \
    --cache_dir "/moai/cache" \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_small.txt \
    --ae CausalVAEModel_4x8x8 \
    --save_img_path "./sample_video_raw_65x512x512" \
    --fps 24 \
    --guidance_scale 1.0 \
    --num_sampling_steps 150 \
    --enable_tiling
    #--model_path /moai/model/opensora_overfit/dummy/final/1_4501/model/pytorch_model.bin \
    #--model_path /moai/model/opensora_overfit_17f/32_212/model/pytorch_model.bin \
    #--model_path "/moai/model/65x512x512_10node_bs2_lr2e-5_4img/checkpoint-18000" \
    #--num_frames 17 \

# /moai/cache/models--LanguageBind--Open-Sora-Plan-v1.1.0/snapshots/4d6a61ab745d7499236d8650b21676aae71271a4/65x512x512/diffusion_pytorch_model.safetensors
