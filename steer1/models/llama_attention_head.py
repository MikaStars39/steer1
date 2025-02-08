import torch
from transformers.cache_utils import Cache
from typing import Optional, Tuple, List
from .utils import intervene_state

def llama_model_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        from ..steer.steer import steer
        FUNCTION = steer(steer_method=self.steering_method)
        hidden_states = FUNCTION(hidden_states, self.vectors, self.coefficient)
 
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs

def intervene_llama_layer(
    model: torch.nn.Module, 
    vectors: List[torch.Tensor], 
    layer_ids: List[int], 
    steering_method: str, 
    coefficient: float,
):

    """
    Intervene attention heads with Flash Attention support for LLaMA models.
    
    Args:
        model: The LLaMA transformer model
        vectors: List of vectors
    """
    # Replace forward methods of attention layers

    for i in range(len(model.model.layers)):
        if i in layer_ids:
            model.model.layers[i].forward = llama_model_forward.__get__(
                model.model.layers[i]
            )
            model.model.layers[i].vectors = vectors[i]
            model.model.layers[i].steering_method = steering_method
            model.model.layers[i].coefficient = coefficient
        
    return model