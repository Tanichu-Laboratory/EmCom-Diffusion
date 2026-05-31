"""
Seq2Seq Transformer: EC tokens → NL caption  (Yao et al. 2022, Section 5)

Architecture:
  Encoder: Embedding(ec_vocab, d_model) + positional + 3 TransformerEncoder layers
  Decoder: Embedding(nl_vocab, d_model) + positional + 6 TransformerDecoder layers
  Output:  Linear(d_model, nl_vocab)  [weight-tied to decoder embedding]
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class ECToNLTranslator(nn.Module):
    def __init__(
        self,
        ec_vocab_size: int,   # 256
        nl_vocab_size: int,   # GPT2: 50257
        d_model: int = 256,
        nhead: int = 8,
        n_enc_layers: int = 3,
        n_dec_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        ec_pad_id: int = 256,
        nl_pad_id: int = 50256,
    ):
        super().__init__()
        self.d_model = d_model
        self.nl_vocab_size = nl_vocab_size
        self.nl_pad_id = nl_pad_id

        # encoder side
        self.ec_embed = nn.Embedding(ec_vocab_size + 1, d_model, padding_idx=ec_pad_id)
        self.enc_pos  = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers,
                                             enable_nested_tensor=False)

        # decoder side
        self.nl_embed = nn.Embedding(nl_vocab_size, d_model, padding_idx=nl_pad_id)
        self.dec_pos  = PositionalEncoding(d_model, dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # output projection (weight-tied with nl_embed)
        self.out_proj = nn.Linear(d_model, nl_vocab_size, bias=False)
        self.out_proj.weight = self.nl_embed.weight

        self._init_weights()

    def _init_weights(self):
        # only init non-embedding linear weights
        for name, p in self.named_parameters():
            if "embed" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, ec_tokens: torch.Tensor) -> torch.Tensor:
        """ec_tokens: [B, K]  →  memory: [B, K, d_model]"""
        src_key_padding_mask = (ec_tokens == self.ec_embed.padding_idx)
        x = self.enc_pos(self.ec_embed(ec_tokens))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask), src_key_padding_mask

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        tgt: [B, T]  →  logits: [B, T, nl_vocab]
        """
        T = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)
        x = self.dec_pos(self.nl_embed(tgt))
        out = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)  # [B, T, nl_vocab]

    def forward(self, ec_tokens: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Training forward: teacher-forcing.  Returns logits [B, T, nl_vocab]."""
        memory, mem_pad_mask = self.encode(ec_tokens)
        return self.decode(tgt, memory, mem_pad_mask)

    @torch.no_grad()
    def sample_decode(
        self,
        ec_tokens: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> list[list[int]]:
        """
        Nucleus (top-p) sampling decode.
        Returns list of token-id lists (no BOS/EOS).
        """
        B = ec_tokens.size(0)
        device = ec_tokens.device
        memory, mem_pad = self.encode(ec_tokens)

        tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        results = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decode(tgt, memory, mem_pad)[:, -1, :]  # [B, V]
            logits = logits / temperature

            # top-p filtering
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(-1).cumsum(-1)
            # remove tokens after cumulative prob exceeds top_p
            remove_mask = cumprobs - sorted_logits.softmax(-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            # scatter back
            logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

            for i in range(B):
                if not done[i]:
                    if next_id[i].item() == eos_id:
                        done[i] = True
                    else:
                        results[i].append(next_id[i].item())
            if done.all():
                break
            tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)

        return results

    @torch.no_grad()
    def greedy_decode(
        self,
        ec_tokens: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 50,
    ) -> list[list[int]]:
        """Greedy decode for inference.  Returns list of token-id lists (no BOS)."""
        B = ec_tokens.size(0)
        device = ec_tokens.device
        memory, mem_pad = self.encode(ec_tokens)

        tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        results = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decode(tgt, memory, mem_pad)  # [B, t, V]
            next_id = logits[:, -1, :].argmax(-1)       # [B]
            for i in range(B):
                if not done[i]:
                    if next_id[i].item() == eos_id:
                        done[i] = True
                    else:
                        results[i].append(next_id[i].item())
            if done.all():
                break
            tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)

        return results
