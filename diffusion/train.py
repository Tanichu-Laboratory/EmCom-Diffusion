"""
Fine-tune Stable Diffusion v1.5 with emergent-language text conditioning.

Settings A / B / C are controlled by a YAML config (--config):
  A: EC embedding only (U-Net fully frozen)
  B: EC embedding + U-Net LoRA
  C: EC embedding + CLIP top-4 layers + U-Net LoRA

Usage:
  accelerate launch -m diffusion.train --config configs/setting_b.yaml
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import sys

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from data import build_datasets
from ec_text_encoder import build_ec_text_encoder, count_trainable
from utils import load_config, encode_images, vae_to_precision

COCO_ROOT = Path(os.environ.get("COCO_ROOT", "data/coco2017"))
TRAIN_ANN = COCO_ROOT / "annotations" / "captions_train2017.json"
VAL_ANN   = COCO_ROOT / "annotations" / "captions_val2017.json"
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(_HERE.parent)))


def _pad_to_clip_len(ec_tokens: torch.Tensor, attention_mask: torch.Tensor,
                     clip_seq_len: int = 77, pad_id: int = 0):
    """Pad/truncate (B,T) tensors to clip_seq_len=77."""
    B, T = ec_tokens.shape
    if T == clip_seq_len:
        return ec_tokens, attention_mask
    if T > clip_seq_len:
        return ec_tokens[:, :clip_seq_len], attention_mask[:, :clip_seq_len]
    # pad
    pad_len = clip_seq_len - T
    ids  = torch.full((B, pad_len), pad_id, dtype=torch.long, device=ec_tokens.device)
    mask = torch.zeros((B, pad_len), dtype=torch.long, device=ec_tokens.device)
    return (torch.cat([ec_tokens, ids], dim=1),
            torch.cat([attention_mask, mask], dim=1))



def save_checkpoint(accelerator, unet, text_enc, cfg, step, ckpt_dir):
    """Save LoRA weights and custom text encoder embedding."""
    ckpt_path = Path(ckpt_dir) / f"step_{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_enc  = accelerator.unwrap_model(text_enc)

    if cfg.get("unet_lora_rank", 0) > 0:
        unwrapped_unet.save_pretrained(str(ckpt_path / "unet_lora"))

    # always save custom token embedding (and optionally unfrozen CLIP layers)
    torch.save(
        unwrapped_enc.state_dict(),
        str(ckpt_path / "text_encoder_state.pt"),
    )
    # save config alongside checkpoint
    with open(ckpt_path / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f)
    return str(ckpt_path)


def validate_and_log(accelerator, unet, text_enc, vae, val_scheduler,
                     val_batch, cfg, step, output_dir):
    """Quick val: CFG denoising on a small fixed batch → save images + optional wandb."""
    if not accelerator.is_main_process:
        return

    device   = accelerator.device
    pad_id   = cfg["pad_token_id"]
    clip_len = 77
    guidance = cfg.get("guidance_scale", 7.5)

    ids, mask = _pad_to_clip_len(
        val_batch["ec_tokens"].to(device),
        val_batch["attention_mask"].to(device),
        clip_len, pad_id,
    )

    unwrap_enc  = accelerator.unwrap_model(text_enc)
    unwrap_unet = accelerator.unwrap_model(unet)

    with torch.no_grad():
        cond_hidden = unwrap_enc(input_ids=ids, attention_mask=mask).last_hidden_state

        B = ids.shape[0]
        if guidance > 1.0:
            uncond_ids  = torch.full((B, clip_len), pad_id, dtype=torch.long, device=device)
            uncond_mask = torch.ones(B, clip_len, dtype=torch.long, device=device)
            uncond_hidden  = unwrap_enc(input_ids=uncond_ids, attention_mask=uncond_mask).last_hidden_state
            encoder_hidden = torch.cat([uncond_hidden, cond_hidden])  # [2B, 77, 768]
        else:
            encoder_hidden = cond_hidden

        val_scheduler.set_timesteps(30)
        latents = torch.randn(B, 4, 64, 64, device=device, dtype=cond_hidden.dtype)

        for t in val_scheduler.timesteps:
            if guidance > 1.0:
                noise_pred_raw = unwrap_unet(torch.cat([latents, latents]), t, encoder_hidden).sample
                noise_uncond, noise_cond = noise_pred_raw.chunk(2)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = unwrap_unet(latents, t, encoder_hidden).sample
            latents = val_scheduler.step(noise_pred, t, latents).prev_sample

        vae_dtype = next(vae.parameters()).dtype
        images = vae.decode((latents / vae.config.scaling_factor).to(vae_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)

    out_dir = Path(output_dir) / f"val_step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pil_images = []
    for i, img_t in enumerate(images):
        img_np = (img_t.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil.save(str(out_dir / f"sample_{i:02d}.png"))
        pil_images.append(pil)

    if cfg.get("log_with") == "wandb":
        import wandb
        accelerator.log({"val_images": [wandb.Image(p) for p in pil_images]}, step=step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=cfg.get("grad_accumulation", 4),
        log_with=cfg.get("log_with", None),
        project_dir=str(PROJECT_DIR / "outputs"),
    )

    if accelerator.is_main_process and cfg.get("log_with") == "wandb":
        accelerator.init_trackers(
            project_name=cfg.get("wandb_project", "pilot_ec2img"),
            config=cfg,
            init_kwargs={"wandb": {"name": cfg.get("run_name", "run")}},
        )

    # ── Load models ──────────────────────────────────────────
    base_model  = cfg["base_model"]
    noise_sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    vae         = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet        = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")

    text_enc = build_ec_text_encoder(
        vocab_size=cfg["vocab_size"],
        pad_token_id=cfg["pad_token_id"],
        unfreeze_top_k_layers=cfg.get("unfreeze_top_k_layers", 0),
    )

    # VAE: always frozen
    vae.requires_grad_(False)

    # U-Net: frozen (setting A) or LoRA (settings B & C)
    unet_lora_rank = cfg.get("unet_lora_rank", 0)
    if unet_lora_rank > 0:
        lora_cfg = LoraConfig(
            r=unet_lora_rank,
            lora_alpha=unet_lora_rank,
            target_modules=cfg.get("lora_target_modules",
                                   ["to_q", "to_k", "to_v", "to_out.0"]),
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_cfg)
        unet.print_trainable_parameters()
    else:
        unet.requires_grad_(False)

    if cfg.get("gradient_checkpointing", False):
        unet.enable_gradient_checkpointing()

    # ── Optimizer ────────────────────────────────────────────
    lr_embed = cfg.get("lr_embedding", 1e-4)
    lr_lora  = cfg.get("lr_lora", 1e-4)
    lr_clip  = cfg.get("lr_clip_unfrozen", 5e-5)

    # group params with separate learning rates
    param_groups = []
    embed_params = list(text_enc.text_model.embeddings.token_embedding.parameters())
    embed_ids    = {id(p) for p in embed_params}

    clip_unfrozen = [
        p for p in text_enc.parameters()
        if p.requires_grad and id(p) not in embed_ids
    ]
    unet_params = [p for p in unet.parameters() if p.requires_grad]

    if embed_params:
        param_groups.append({"params": embed_params,   "lr": lr_embed})
    if clip_unfrozen:
        param_groups.append({"params": clip_unfrozen,  "lr": lr_clip})
    if unet_params:
        param_groups.append({"params": unet_params,    "lr": lr_lora})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-2))

    # ── Data ─────────────────────────────────────────────────
    ec_json = str(PROJECT_DIR / "outputs" / cfg.get("ec_json", "ec_captions.json"))
    train_ds, val_ds = build_datasets(
        ec_json, str(TRAIN_ANN), str(VAL_ANN),
        image_size=cfg.get("image_size", 512),
    )

    # small subset mode (for quick loop testing)
    subset_size = cfg.get("train_subset_size", None)
    if subset_size is not None:
        indices = random.sample(range(len(train_ds)), min(subset_size, len(train_ds)))
        train_ds = Subset(train_ds, indices)

    batch_size = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
    )

    # fixed val batch (same across settings for fair comparison)
    val_seed = cfg.get("seed", 42)
    rng = random.Random(val_seed)
    val_indices = rng.sample(range(len(val_ds)), min(8, len(val_ds)))
    val_batch_list = [val_ds[i] for i in val_indices]
    val_batch = {
        k: torch.stack([b[k] for b in val_batch_list])
        if isinstance(val_batch_list[0][k], torch.Tensor) else
        [b[k] for b in val_batch_list]
        for k in val_batch_list[0]
    }

    # ── LR scheduler ─────────────────────────────────────────
    total_steps   = cfg.get("total_steps", 10000)
    warmup_steps  = cfg.get("warmup_steps", 500)
    grad_accum    = cfg.get("grad_accumulation", 4)

    lr_sched = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ───────────────────────────────────
    unet, text_enc, optimizer, train_loader, lr_sched = accelerator.prepare(
        unet, text_enc, optimizer, train_loader, lr_sched
    )

    # Compute after prepare so len(train_loader) reflects per-process size
    num_update_steps = math.ceil(len(train_loader) / grad_accum)
    num_epochs = math.ceil(total_steps / num_update_steps)
    vae = vae.to(accelerator.device)
    vae = vae_to_precision(vae, cfg.get("mixed_precision", "fp16"))

    # ── Training loop ─────────────────────────────────────────
    global_step   = 0
    pad_id        = cfg["pad_token_id"]
    clip_len      = 77
    ckpt_dir      = str(PROJECT_DIR / "checkpoints" / cfg.get("run_name", "pilot"))
    output_dir    = str(PROJECT_DIR / "outputs" / cfg.get("run_name", "pilot"))
    val_interval  = cfg.get("val_interval", 500)
    ckpt_interval = cfg.get("ckpt_interval", 2500)

    val_scheduler = DDIMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")

    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process)

    for epoch in range(num_epochs):
        unet.train()
        text_enc.train()

        for batch in train_loader:
            if global_step >= total_steps:
                break

            with accelerator.accumulate(unet, text_enc):
                images     = batch["image"].to(accelerator.device)
                ec_tokens  = batch["ec_tokens"].to(accelerator.device)
                attn_mask  = batch["attention_mask"].to(accelerator.device)

                # encode image to latent
                latents = encode_images(vae, images.to(vae.dtype))
                latents = latents.to(accelerator.device)

                # sample noise + timesteps
                noise = torch.randn_like(latents)
                B     = latents.shape[0]
                ts    = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (B,), device=accelerator.device).long()
                noisy = noise_sched.add_noise(latents, noise, ts)

                # text encoding
                ids, mask = _pad_to_clip_len(ec_tokens, attn_mask, clip_len, pad_id)
                enc_out   = text_enc(input_ids=ids, attention_mask=mask)
                enc_hidden = enc_out.last_hidden_state

                # diffusion loss
                noise_pred = unet(noisy, ts, enc_hidden).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in list(text_enc.parameters()) + list(unet.parameters())
                         if p.requires_grad],
                        cfg.get("max_grad_norm", 1.0),
                    )
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

                if accelerator.is_main_process:
                    logs = {"loss": loss.item(), "lr": lr_sched.get_last_lr()[0]}
                    accelerator.log(logs, step=global_step)

                if global_step % val_interval == 0:
                    unet.eval(); text_enc.eval()
                    validate_and_log(
                        accelerator, unet, text_enc, vae, val_scheduler,
                        val_batch, cfg, global_step, output_dir,
                    )
                    unet.train(); text_enc.train()

                if global_step % ckpt_interval == 0:
                    save_checkpoint(accelerator, unet, text_enc, cfg, global_step, ckpt_dir)

            if global_step >= total_steps:
                break

    # final checkpoint
    save_checkpoint(accelerator, unet, text_enc, cfg, global_step, ckpt_dir)
    accelerator.end_training()
    progress.close()
    print(f"Training complete. Final step: {global_step}")


if __name__ == "__main__":
    main()
