import os
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def extract_vectors(
    model: torch.nn.Module, 
    layer_start: int, 
    layer_end: int, 
    dataloader: torch.utils.data.DataLoader, 
    negative_dataloader: torch.utils.data.DataLoader = None, 
    device: str = "cuda", 
    save_path: str = "hidden_states.pt",
):
    '''
    extract hidden states from the model
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    negative_dataloader: torch.utils.data.DataLoader
    layer_start: int, the start layer
    layer_end: int, the end layer
    device: str
    save_path: str
    '''
    model.eval()

    # 1. check if the hidden states are already extracted
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        hidden_states = torch.load(save_path)
        print("Load a saved hidden states from {}".format(save_path))
        print("The shape of the hidden states is {}".format(hidden_states.shape))
        return hidden_states
    
    # 2. extract the hidden states
    hidden_states = []
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        batch_hidden_states = [
            outputs.hidden_states[i][:, -1, :] for i in range(layer_start, layer_end + 1)
        ]
        hidden_states.append(torch.stack(batch_hidden_states, dim=0))
        
    # 3. extract the negative hidden states (in difference in means)
    if negative_dataloader is not None:
        negative_hidden_states = []
        for batch in tqdm(negative_dataloader):
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            batch_hidden_states = [
                outputs.hidden_states[i][:, -1, :] for i in range(layer_start, layer_end + 1)
            ]
            negative_hidden_states.append(torch.stack(batch_hidden_states, dim=0))

    # hidden states: layer, batch_size, d

    # 4. average the hidden states
    hidden_states = torch.cat(hidden_states, dim=1)
    hidden_states = torch.mean(hidden_states, dim=1)
    if negative_dataloader is not None:
        negative_hidden_states = torch.cat(negative_hidden_states, dim=1)
        negative_hidden_states = torch.mean(negative_hidden_states, dim=1)
        hidden_states = hidden_states - negative_hidden_states # difference in means
    
    # 5. save the hidden states
    torch.save(hidden_states, save_path)
    print("Save a saved hidden states to {}".format(save_path))
    print("The shape of the hidden states is {}".format(hidden_states.shape))
    return hidden_states
