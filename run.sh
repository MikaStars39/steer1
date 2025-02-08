cd /mnt/weka/hw_workspace/qy_workspace/steer
XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache CUDA_VISIBLE_DEVICES=7 python steer.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_name_or_path "JailbreakBench/JBB-Behaviors" \
    --vector_path "/mnt/weka/hw_workspace/qy_workspace/steer/dataset/cad_sentiment_hidden_states.pt" \
    --steering_method "caa" \
    --coefficient "1" \
    --batch_size 4 \
    --layer_start 12 \
    --layer_end 24 \
    --max_new_tokens 10240 \
    --token "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"