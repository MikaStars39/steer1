XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache CUDA_VISIBLE_DEVICES=7 \
    python preprocess.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_name_or_path "/mnt/weka/hw_workspace/qy_workspace/steer/dataset/cad_sentiment.tsv" \
    --batch_size 8 \
    --layer_start 0 \
    --layer_end 32 \
    --token "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" \
    --save_path "/mnt/weka/hw_workspace/qy_workspace/steer/dataset/cad_sentiment_hidden_states.pt"