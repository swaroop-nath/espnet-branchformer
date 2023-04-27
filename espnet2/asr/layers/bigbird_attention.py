import torch.nn as nn
import torch
from transformers import BigBirdConfig, BigBirdModel
from transformers.models.big_bird.modeling_big_bird import BigBirdAttention

class LinearRandomAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attn_heads, attn_type, block_size, num_random_blocks, attn_dropout_rate):
        super().__init__()
        self.bigbird_config = BigBirdConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attn_heads,
            attention_probs_dropout_prob=attn_dropout_rate,
            attention_type=attn_type,
            block_size=block_size,
            num_random_blocks=num_random_blocks
        )
        self.attn_layer = BigBirdAttention(self.bigbird_config)
        self.block_size = block_size

    def forward(self, xs_pad, mask):
        """Forward method.

        Args:
            xs_pad: (batch, time, size = n_heads * attn_dim)
            mask: (batch, 1, time), nonpadding is 1, padding is 0

        Returns:
            torch.Tensor: (batch, time, size)
        """
        mask = mask.squeeze(dim=1).to(torch.int)
        padding_len, xs_pad, mask = self._pad_to_block_size(xs_pad, mask, self.bigbird_config.pad_token_id)
        
        blocked_encoder_mask, band_mask, from_mask, to_mask = BigBirdModel.create_masks_for_block_sparse_attn(mask, self.block_size)
        self.attn_layer.set_attention_type("block_sparse")
        attn_output = self.attn_layer(hidden_states=xs_pad, attention_mask=mask, band_mask=band_mask, from_mask=from_mask,
                            to_mask=to_mask, from_blocked_mask=blocked_encoder_mask, to_blocked_mask=blocked_encoder_mask)
        
        attn_weighted_vectors = attn_output[0]
        if padding_len > 0: attn_weighted_vectors = attn_weighted_vectors[:, :-padding_len, :]
        return attn_weighted_vectors
    
    def _pad_to_block_size(
        self, 
        xs_pad: torch.Tensor,
        mask: torch.Tensor,
        pad_token_id: int,
    ): # works on token ids, not hidden representation -- need to make this right
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # padding
        block_size = self.bigbird_config.block_size

        input_shape = xs_pad.size()
        batch_size, seq_len, embed_dim = input_shape

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            pad_tensor = torch.ones((padding_len, embed_dim), device=xs_pad.device) * pad_token_id
            pad_tensor = pad_tensor.expand(batch_size, padding_len, embed_dim)
            xs_pad = torch.cat((xs_pad, pad_tensor), dim=1)

            assert xs_pad.size() == (batch_size, (seq_len + padding_len), embed_dim)

            mask_pad_tensor = torch.ones(padding_len, device=xs_pad.device) * pad_token_id
            mask_pad_tensor = mask_pad_tensor.expand(batch_size, padding_len)
            mask = torch.cat((mask, mask_pad_tensor), dim=1)
            
            assert mask.size() == (batch_size, (seq_len + padding_len))

        return padding_len, xs_pad, mask
