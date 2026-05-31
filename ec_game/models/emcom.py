"""
EmCom-Diffusion: Referential Game model.

Speaker:
  K learnable query vectors → cross-attention to frozen DINOv2 ViT-B/14 patch tokens
  → linear projection to logits → K discrete tokens via straight-through Gumbel-softmax

Receiver:
  K tokens (soft one-hot) → BERT text encoder (6-layer, hidden=768)
  → L2-normalised projection → 256-dim text feature

Training objective:
  InfoNCE contrastive loss between image feature and text feature.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ec_game.models.blip_utils import create_vit
from ec_game.models.med import BertConfig, BertModel


def gumbel_softmax_st(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Straight-through Gumbel-softmax. logits: [B, K, V] → [B, K, V]."""
    soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
    return (hard - soft).detach() + soft


class SimpleEmCom(nn.Module):
    def __init__(
        self,
        med_config: str,
        image_size: int = 224,
        vit: str = 'base',
        vit_grad_ckpt: bool = False,
        vit_ckpt_layer: int = 0,
        vocab_size: int = 256,
        num_slots: int = 8,        # K tokens per message
        embed_dim: int = 256,
        num_attention_heads: int = 6,
        num_hidden_layers: int = 6,
        temp: float = 0.07,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_slots  = num_slots
        self.temp = nn.Parameter(torch.ones([]) * temp)

        # Visual encoder: frozen DINOv2 ViT-B/14
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0
        )
        self.vision_proj = nn.Linear(vision_width, embed_dim)

        # Speaker: K learnable queries cross-attend to DINOv2 patch tokens
        self.sender_queries = nn.Parameter(torch.randn(num_slots, vision_width))
        nn.init.xavier_uniform_(self.sender_queries.unsqueeze(0))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=8,
            batch_first=True,
            dropout=0.0,
        )
        self.to_logits = nn.Linear(vision_width, vocab_size)

        # Receiver: 6-layer BERT text encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.vocab_size          = vocab_size
        encoder_config.num_attention_heads = num_attention_heads
        encoder_config.num_hidden_layers   = num_hidden_layers
        encoder_config.encoder_width       = vision_width
        encoder_config.add_cross_attention  = False
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(vocab_size)

        text_width = encoder_config.hidden_size  # 768
        self.text_proj = nn.Linear(text_width, embed_dim)

    def _encode_visual(self, image: torch.Tensor):
        """Returns (image_embeds [B, N+1, D], image_feat [B, embed_dim])."""
        image_embeds = self.visual_encoder(image)
        image_feat   = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_embeds, image_feat

    def _encode_message(self, soft: torch.Tensor):
        """
        soft [B, K, vocab_size] → (text_feat [B, embed_dim], hidden [B, K+1, text_width]).
        Differentiable soft lookup: soft @ word_embedding.weight.
        """
        B, K, V  = soft.shape
        device   = soft.device
        word_emb = self.text_encoder.embeddings.word_embeddings

        token_embeds    = soft @ word_emb.weight              # [B, K, text_width]
        cls_emb         = word_emb(torch.zeros(B, 1, dtype=torch.long, device=device))
        inputs_embeds   = torch.cat([cls_emb, token_embeds], dim=1)  # [B, K+1, text_width]
        attn_mask       = torch.ones(B, K + 1, dtype=torch.long, device=device)

        out = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            return_dict=True,
            mode='text',
        )
        text_feat = F.normalize(self.text_proj(out.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat, out.last_hidden_state

    def forward(self, image: torch.Tensor, tau: float = 1.0):
        """
        Returns:
            loss_ref   : InfoNCE contrastive loss
            loss_recon : always 0 (kept for API compatibility with train.py)
            tokens     : [B, K] hard token IDs (for logging)
            token_ent  : mean entropy of raw token distribution (for regularisation)
        """
        image_embeds, image_feat = self._encode_visual(image)

        # Speaker: cross-attention → Gumbel-softmax
        B       = image.size(0)
        patches = image_embeds[:, 1:, :]          # exclude CLS token
        queries = self.sender_queries.unsqueeze(0).expand(B, -1, -1)
        slots, _ = self.cross_attn(queries, patches, patches)
        logits   = self.to_logits(slots)          # [B, K, vocab_size]

        if self.training:
            soft = gumbel_softmax_st(logits, tau)
        else:
            soft = F.one_hot(logits.argmax(-1), self.vocab_size).float()

        # Receiver
        text_feat, _ = self._encode_message(soft)

        # InfoNCE
        labels   = torch.arange(B, device=image.device)
        sim_i2t  = image_feat @ text_feat.T / self.temp
        sim_t2i  = text_feat  @ image_feat.T / self.temp
        loss_ref = (F.cross_entropy(sim_i2t, labels) +
                    F.cross_entropy(sim_t2i, labels)) / 2

        loss_recon = image.new_zeros(1).squeeze()
        raw_probs  = logits.softmax(-1)
        token_ent  = -(raw_probs * raw_probs.clamp(min=1e-8).log()).sum(-1).mean()

        return loss_ref, loss_recon, logits.argmax(-1), token_ent

    def train(self, mode: bool = True):
        super().train(mode)
        # DINOv2 is frozen; keep it in eval mode to disable DropPath stochasticity.
        self.visual_encoder.model.eval()
        return self

    @torch.no_grad()
    def get_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Inference: [B, 3, H, W] → [B, K] token IDs."""
        image_embeds = self.visual_encoder(image)
        patches  = image_embeds[:, 1:, :]
        B        = image.size(0)
        queries  = self.sender_queries.unsqueeze(0).expand(B, -1, -1)
        slots, _ = self.cross_attn(queries, patches, patches)
        return self.to_logits(slots).argmax(-1)


def build_model(config: dict) -> SimpleEmCom:
    med_config = os.path.join(os.path.dirname(__file__), '..', 'configs', 'bert_config.json')
    return SimpleEmCom(
        med_config          = med_config,
        image_size          = config.get('image_size', 224),
        vit                 = config.get('vit', 'base'),
        vit_grad_ckpt       = config.get('vit_grad_ckpt', False),
        vit_ckpt_layer      = config.get('vit_ckpt_layer', 0),
        vocab_size          = config.get('vocab_size', 256),
        num_slots           = config.get('num_slots', 8),
        embed_dim           = config.get('embed_dim', 256),
        num_attention_heads = config.get('num_attention_heads', 6),
        num_hidden_layers   = config.get('num_hidden_layers', 6),
        temp                = config.get('temp', 0.07),
    )
