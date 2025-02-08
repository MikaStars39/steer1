# -*- coding: utf-8 -*-

import json
import fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
import torch
import os

from steer1.intervene.extraction import extract_vectors

def preprocess(
    model_name_or_path: str,
    dataset_name_or_path: str,
    batch_size: int,
    layer_start: int,
    layer_end: int,
    token: str,
    save_path: str,
):
    '''
    Preprocesses data for model training or evaluation.

    This function loads a pre-trained model and a dataset, tokenizes the prompts,
    and extracts hidden states from specified layers of the model. It supports
    both positive and negative prompts for certain datasets.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model to load.
        dataset_name_or_path (str): The name or path of the dataset to use.
        batch_size (int): The batch size for data loading.
        layer_start (int): The starting layer for hidden state extraction.
        layer_end (int): The ending layer for hidden state extraction.
        token (str): The Hugging Face token for authentication.
        save_path (str): The path to save the extracted hidden states.

    Raises:
        ValueError: If the dataset name or path is not supported.

    Note:
        The function uses CUDA if available, otherwise falls back to CPU.
    '''
    # 1. preparations:
    # login with huggingface
    from huggingface_hub import login
    login(token=token)

    # cuda device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2.load the model and tokenizer
    print("model_name_or_path", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')

    negative_prompts = None

    # 3. get the dataset
    if "test" in dataset_name_or_path:
        from dataset.dataset import prepare_ours_jailbreak_dataset
        positive_prompts = prepare_ours_jailbreak_dataset(path=dataset_name_or_path)
    elif "cad" in dataset_name_or_path:
        from dataset.dataset import prepare_cad_dataset
        positive_prompts, negative_prompts = prepare_cad_dataset(path=dataset_name_or_path)
    else:
        raise ValueError("Dataset name or path not supported")
        
    # 4. tokenize the prompts and datasets
    tokenized_prompts = tokenizer(positive_prompts, padding=True, truncation=True, return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(tokenized_prompts.input_ids, tokenized_prompts.attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if negative_prompts is not None:
        tokenized_prompts = tokenizer(negative_prompts, padding=True, truncation=True, return_tensors="pt")
        negative_dataset = torch.utils.data.TensorDataset(tokenized_prompts.input_ids, tokenized_prompts.attention_mask)
        negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=False)
      
    # 5. extract the hidden states
    hidden_states = extract_vectors(
        model, 
        dataloader=dataloader, 
        negative_dataloader=negative_dataloader, 
        layer_start=layer_start, 
        layer_end=layer_end, 
        device=device,
        save_path=save_path,
    )

if __name__ == "__main__":
    # use fire
    fire.Fire(preprocess)