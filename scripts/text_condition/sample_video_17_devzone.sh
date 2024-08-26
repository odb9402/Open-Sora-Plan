export PYTHONPATH=$PYTHONPATH:~/Open-Sora-Plan
MOREH_USE_MODNN_RANDOM_GENERATOR=0 \
python opensora/sample/sample_t2v.py \
    --model_path "/home/share/opensora/models--LanguageBind--Open-Sora-Plan-v1.1.0/snapshots/4d6a61ab745d7499236d8650b21676aae71271a4/65x512x512/diffusion_pytorch_model.pt" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --version 65x512x512 \
    --num_frames 17 \
    --height 512 \
    --width 512 \
    --cache_dir "/home/share/opensora" \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_other_prompt.txt \
    --ae CausalVAEModel_4x8x8 \
    --save_img_path "./sample_video_raw_17x512x512_modnn0_2" \
    --fps 24 \
    --guidance_scale 7.0 \
    --num_sampling_steps 150 \
    --enable_tiling