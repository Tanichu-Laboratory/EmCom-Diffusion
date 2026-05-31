#!/usr/bin/env python3
"""
Experiment A1 (Step 1): Multi-seed image generation.

For each of 500 selected val samples, generate N=8 images with different random
seeds under each condition (EC, Random, Fixed, NL_ft).

Output layout:
    outputs/experiments_v2/multiseed/{condition}/{sample_idx:04d}_seed{seed:02d}.png
    outputs/experiments_v2/multiseed/metadata.json  ← maps sample_idx → image_id, ec_tokens, gt_path

Usage:
    cd .
    python experiments_v2/exp_a1_generate_multiseed.py \
        --n_samples 500 --n_seeds 8 \
        --conditions ec random fixed nl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from PIL import Image
from tqdm.auto import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / '..' / 'diffusion'))


def _pad_to_77(ec_tokens: torch.Tensor, attention_mask: torch.Tensor,
               pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = ec_tokens.shape
    if T >= 77:
        return ec_tokens[:, :77], attention_mask[:, :77]
    pad = 77 - T
    ids = torch.full((B, pad), pad_id, dtype=torch.long, device=ec_tokens.device)
    mask = torch.zeros((B, pad), dtype=torch.long, device=ec_tokens.device)
    return torch.cat([ec_tokens, ids], 1), torch.cat([attention_mask, mask], 1)


@torch.no_grad()
def generate_single(text_enc, unet, vae, scheduler,
                    tokens: list[int], pad_id: int,
                    seed: int, guidance: float, n_steps: int,
                    device: str) -> Image.Image:
    """Generate one image from a token list with a given seed."""
    torch.manual_seed(seed)

    T = len(tokens)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)       # [1, T]
    mask = torch.ones(1, T, dtype=torch.long, device=device)
    ids, mask = _pad_to_77(ids, mask, pad_id)

    cond = text_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    if guidance > 1.0:
        uncond_ids = torch.full((1, 77), pad_id, dtype=torch.long, device=device)
        uncond_mask = torch.ones(1, 77, dtype=torch.long, device=device)
        uncond = text_enc(input_ids=uncond_ids, attention_mask=uncond_mask).last_hidden_state
        encoder_hidden = torch.cat([uncond, cond])
    else:
        encoder_hidden = cond

    latents = torch.randn(1, 4, 64, 64, device=device, dtype=cond.dtype)
    scheduler.set_timesteps(n_steps)
    for t in scheduler.timesteps:
        if guidance > 1.0:
            raw = unet(torch.cat([latents, latents]), t, encoder_hidden).sample
            uncond_pred, cond_pred = raw.chunk(2)
            noise = uncond_pred + guidance * (cond_pred - uncond_pred)
        else:
            noise = unet(latents, t, encoder_hidden).sample
        latents = scheduler.step(noise, t, latents).prev_sample

    img = vae.decode(latents / vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


@torch.no_grad()
def generate_nl_single(caption: str, tokenizer, text_enc, unet, vae, scheduler,
                       seed: int, guidance: float, n_steps: int,
                       device: str) -> Image.Image:
    """Generate one NL-conditioned image."""
    torch.manual_seed(seed)

    enc = tokenizer([caption], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").to(device)
    cond = text_enc(**enc).last_hidden_state

    if guidance > 1.0:
        uncond_enc = tokenizer([""], padding="max_length", max_length=77,
                               return_tensors="pt").to(device)
        uncond = text_enc(**uncond_enc).last_hidden_state
        encoder_hidden = torch.cat([uncond, cond])
    else:
        encoder_hidden = cond

    latents = torch.randn(1, 4, 64, 64, device=device, dtype=cond.dtype)
    scheduler.set_timesteps(n_steps)
    for t in scheduler.timesteps:
        if guidance > 1.0:
            raw = unet(torch.cat([latents, latents]), t, encoder_hidden).sample
            uncond_pred, cond_pred = raw.chunk(2)
            noise = uncond_pred + guidance * (cond_pred - uncond_pred)
        else:
            noise = unet(latents, t, encoder_hidden).sample
        latents = scheduler.step(noise, t, latents).prev_sample

    img = vae.decode(latents / vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for sample selection and token randomization")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--conditions", nargs="+",
                        default=["ec", "random", "fixed", "nl"],
                        choices=["ec", "random", "fixed", "nl"])
    parser.add_argument("--out_dir",   required=True, help="Output directory for generated images")
    parser.add_argument("--ec_json",   required=True, help="EC corpus JSON (outputs of extract_corpus)")
    parser.add_argument("--ec_ckpt",   required=True, help="SD checkpoint dir for EC condition (pilot_A/step_010000)")
    parser.add_argument("--nl_ckpt",   default=None,  help="SD checkpoint dir for NL condition (pilot_NL/step_010000)")
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--val_ann",   default=None,  help="Path to captions_val2017.json (required for nl condition)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    EC_JSON   = args.ec_json
    EC_CKPT   = Path(args.ec_ckpt)
    NL_CKPT   = Path(args.nl_ckpt) if args.nl_ckpt else None
    BASE_MODEL = args.base_model
    VAL_ANN   = Path(args.val_ann) if args.val_ann else None

    # ── Load EC corpus and sample val entries ──────────────────────────────────
    with open(EC_JSON) as f:
        ec = json.load(f)
    rng = random.Random(args.seed)
    val_all = ec["val"]
    val_entries = rng.sample(val_all, min(args.n_samples, len(val_all)))
    seq_len = ec["num_tokens"]
    vocab_size = ec["vocab_size"]   # 256 (IDs 0-255)
    pad_id = ec["pad_token_id"]     # 256

    # Store metadata (sample_idx → image_id, ec_tokens, gt_path)
    metadata = [
        {"sample_idx": i, "image_id": e["image_id"],
         "ec_tokens": e["ec_tokens"], "gt_path": e["image_path"]}
        for i, e in enumerate(val_entries)
    ]
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"n_samples": len(val_entries), "n_seeds": args.n_seeds,
                   "vocab_size": vocab_size, "pad_id": pad_id,
                   "samples": metadata}, f, indent=2)
    print(f"Saved metadata: {out_dir / 'metadata.json'}")

    # Fixed token sequence (same for all samples)
    fixed_tokens = val_entries[0]["ec_tokens"]

    # ── Load shared VAE + scheduler ────────────────────────────────────────────
    print("Loading VAE and scheduler ...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    # ── Seed list for N seeds ─────────────────────────────────────────────────
    gen_seeds = list(range(args.n_seeds))

    # ─── EC / Random / Fixed conditions ───────────────────────────────────────
    ec_conditions = [c for c in args.conditions if c in ("ec", "random", "fixed")]
    if ec_conditions:
        print("Loading EC text encoder (pilot_A) ...")
        with open(EC_CKPT / "train_config.yaml") as f:
            cfg = yaml.safe_load(f)

        text_enc = build_ec_text_encoder(
            vocab_size=cfg["vocab_size"],
            pad_token_id=cfg["pad_token_id"],
            unfreeze_top_k_layers=cfg.get("unfreeze_top_k_layers", 0),
        )
        state = torch.load(str(EC_CKPT / "text_encoder_state.pt"), map_location="cpu")
        text_enc.load_state_dict(state)
        text_enc = text_enc.to(device).eval()

        unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device).eval()

        _sample_rng = random.Random(args.seed + 1)

        for cond in ec_conditions:
            cond_dir = out_dir / cond
            cond_dir.mkdir(exist_ok=True)
            print(f"\n=== Condition: {cond} ===")

            for entry_meta in tqdm(metadata, desc=cond):
                idx = entry_meta["sample_idx"]
                entry = val_entries[idx]

                if cond == "ec":
                    tokens = entry["ec_tokens"]
                elif cond == "random":
                    tokens = [_sample_rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
                else:  # fixed
                    tokens = fixed_tokens

                for seed in gen_seeds:
                    out_path = cond_dir / f"{idx:04d}_seed{seed:02d}.png"
                    if out_path.exists():
                        continue
                    img = generate_single(
                        text_enc, unet, vae, scheduler,
                        tokens, pad_id, seed,
                        args.guidance, args.n_steps, device,
                    )
                    img.save(str(out_path))

        del text_enc, unet
        torch.cuda.empty_cache()

    # ─── NL condition ─────────────────────────────────────────────────────────
    if "nl" in args.conditions:
        from transformers import CLIPTokenizer, CLIPTextModel
        from peft import PeftModel

        print("\n=== Condition: nl (NL_ft) ===")
        nl_caps = load_nl_captions(str(VAL_ANN))

        tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
        text_enc_nl = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")

        nl_lora_dir = NL_CKPT / "unet_lora"
        if nl_lora_dir.exists():
            unet_base = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
            unet_nl = PeftModel.from_pretrained(unet_base, str(nl_lora_dir)).to(device).eval()
        else:
            unet_nl = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device).eval()
        text_enc_nl = text_enc_nl.to(device).eval()

        nl_dir = out_dir / "nl"
        nl_dir.mkdir(exist_ok=True)

        for entry_meta in tqdm(metadata, desc="nl"):
            idx = entry_meta["sample_idx"]
            image_id = entry_meta["image_id"]
            caption = nl_caps.get(image_id, [""])[0]

            for seed in gen_seeds:
                out_path = nl_dir / f"{idx:04d}_seed{seed:02d}.png"
                if out_path.exists():
                    continue
                img = generate_nl_single(
                    caption, tokenizer, text_enc_nl, unet_nl, vae, scheduler,
                    seed, args.guidance, args.n_steps, device,
                )
                img.save(str(out_path))

        del text_enc_nl, unet_nl
        torch.cuda.empty_cache()

    print(f"\nDone. Images saved to {out_dir}/")


if __name__ == "__main__":
    main()
