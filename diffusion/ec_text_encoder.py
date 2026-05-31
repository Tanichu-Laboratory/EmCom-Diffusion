"""
Custom CLIP text encoder with emergent-language vocabulary.

Replaces the original token embedding of CLIPTextModel with a new one sized
for the emergent vocabulary.  How much of the Transformer is unfrozen is
controlled by `unfreeze_top_k_layers`:
  - 0 → all Transformer layers frozen  (settings A and B)
  - 4 → top-4 layers + final LayerNorm unfrozen (setting C)

The new token embedding is always trainable.
Position embeddings are reused from the pretrained CLIP (sequence ≤ 77).
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel


def build_ec_text_encoder(
    vocab_size: int,
    pad_token_id: int,
    unfreeze_top_k_layers: int = 0,
    pretrained: str = "openai/clip-vit-large-patch14",
) -> CLIPTextModel:
    """
    Args:
        vocab_size:            emergent vocabulary size (e.g. 100 or 101 with pad)
        pad_token_id:          padding token ID for the new embedding
        unfreeze_top_k_layers: number of top CLIP Transformer layers to unfreeze
            0  → all layers frozen   (settings A & B)
            4  → top 4 layers + final LayerNorm unfrozen (setting C)
        pretrained:            HuggingFace model ID for the base CLIP text encoder

    Returns:
        CLIPTextModel with custom token embedding and partial freezing applied.
    """
    clip = CLIPTextModel.from_pretrained(pretrained)

    orig_embed = clip.text_model.embeddings.token_embedding
    new_embed = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=clip.config.hidden_size,  # 768
        padding_idx=pad_token_id,
    )
    with torch.no_grad():
        w = orig_embed.weight
        new_embed.weight.normal_(mean=w.mean().item(), std=w.std().item())
    clip.text_model.embeddings.token_embedding = new_embed

    # freeze all parameters first
    for param in clip.parameters():
        param.requires_grad = False

    # new embedding is always trainable
    clip.text_model.embeddings.token_embedding.weight.requires_grad = True

    # unfreeze top-k Transformer layers
    if unfreeze_top_k_layers > 0:
        layers = clip.text_model.encoder.layers
        for layer in layers[-unfreeze_top_k_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in clip.text_model.final_layer_norm.parameters():
            param.requires_grad = True

    return clip


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    for k in [0, 4]:
        enc = build_ec_text_encoder(vocab_size=101, pad_token_id=100, unfreeze_top_k_layers=k)
        n = count_trainable(enc)
        print(f"unfreeze_top_k_layers={k}: trainable params = {n:,}")

    # forward pass sanity check
    enc = build_ec_text_encoder(vocab_size=101, pad_token_id=100, unfreeze_top_k_layers=0)
    enc.eval()
    B, T = 2, 5
    dummy_ids   = torch.randint(0, 100, (B, T))
    dummy_mask  = torch.ones(B, T, dtype=torch.long)
    # CLIP text encoder expects sequence length up to 77; pad to 77 for position embedding
    SEQ = 77
    input_ids   = torch.full((B, SEQ), 100, dtype=torch.long)
    input_ids[:, :T] = dummy_ids
    attn_mask   = torch.zeros(B, SEQ, dtype=torch.long)
    attn_mask[:, :T] = 1
    with torch.no_grad():
        out = enc(input_ids=input_ids, attention_mask=attn_mask)
    print(f"last_hidden_state shape: {out.last_hidden_state.shape}")  # (2, 77, 768)
    print("OK")
