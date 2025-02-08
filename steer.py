# -*- coding: utf-8 -*-

import json
import fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
from typing import Union, List, Dict
import torch
import os


from steer1.intervene.extraction import extract_vectors
from steer1.models.llama_attention_head import intervene_llama_layer
from huggingface_hub import login

llama_chat_template = "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

def print_summary(model_name, vector_path, steering_method, coefficient, batch_size, layer_start, layer_end, selected_layers, max_new_tokens, device):
    print("=" * 50)
    print("Generation Summary:")
    print(f"Model: {model_name}")
    print(f"Vector Path: {vector_path}")
    print(f"Steering Method: {steering_method}")
    print(f"Coefficient: {coefficient}")
    print(f"Batch Size: {batch_size}")
    print(f"Layers: {layer_start} to {layer_end}")
    print(f"Selected Layers: {selected_layers if selected_layers is not None else 'All'}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Device: {device}")
    print("=" * 50)

def main(
    model_name_or_path: str,
    dataset_name_or_path: str,
    vector_path: str,
    steering_method: str,
    coefficient: float = 0.1,
    batch_size: int = 4,
    layer_start: int = 0,
    layer_end: int = 24,
    selected_layers: int = None,
    max_new_tokens: int = 8,
    token: str = "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",
):
    login(token=token)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_summary(model_name_or_path, vector_path, steering_method, coefficient, batch_size, layer_start, layer_end, selected_layers, max_new_tokens, device)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')

    hidden_states = torch.load(vector_path)
    print(f"Loaded vectors from: {vector_path}")

    layer_ids = list(range(layer_start, layer_end + 1))
    if selected_layers is not None:
        vectors = {i: hidden_states[selected_layers] for i in layer_ids} 
    else:
        vectors = {i: hidden_states[idx] for idx, i in enumerate(layer_ids)}

    model = intervene_llama_layer(
        model, 
        vectors, 
        layer_ids, 
        steering_method, 
        coefficient,
    )

    text = "Write a brief story about a person John and his feelings."
    text = llama_chat_template.format(text)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = tokenizer.decode(outputs[0])
    # save the outputs to a .txt
    with open("outputs.txt", "w") as f:
        f.write(outputs)

if __name__ == "__main__":
    fire.Fire(main)