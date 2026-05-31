"""
Fine-tune SD 1.5 with natural-language (NL) COCO captions as conditioning.

Fair NL baseline: same images and training budget as EC settings, but using
the original CLIP text encoder (frozen) + UNet LoRA rank=16.

Usage:
  accelerate launch --config_file accelerate_configs/1gpu.yaml \
      diffusion/train_nl.py --config configs/setting_nl.yaml
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import sys
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from data import load_nl_captions
from utils import load_config, encode_images, vae_to_precision

PROJECT_DIR = _HERE.parent
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "data/coco2017"))
TRAIN_ANN = COCO_ROOT / "annotations" / "captions_train2017.json"
VAL_ANN   = COCO_ROOT / "annotations" / "captions_val2017.json"


class NLCOCODataset(Dataset):
    """(image, tokenized NL caption) pairs from COCO via an EC corpus JSON."""

    def __init__(self, entries, nl_captions, tokenizer, image_size=512):
        self.entries     = entries
        self.nl_captions = nl_captions
        self.tokenizer   = tokenizer
        self.transform   = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = self.transform(Image.open(entry["image_path"]).convert("RGB"))
        caps  = self.nl_captions.get(entry["image_id"], [""])
        caption = random.choice(caps)
        tok = self.tokenizer(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        return {
            "image":          image,
            "input_ids":      tok.input_ids[0],
            "attention_mask": tok.attention_mask[0],
            "nl_caption":     caption,
        }



def save_checkpoint(accelerator, unet, cfg, step, ckpt_dir):
    ckpt_path = Path(ckpt_dir) / f"step_{step:06d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(str(ckpt_path / "unet_lora"))
    with open(ckpt_path / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f)
    return str(ckpt_path)


def validate_and_log(accelerator, unet, text_enc, vae, val_scheduler,
                     val_batch, cfg, step, output_dir):
    if not accelerator.is_main_process:
        return
    device   = accelerator.device
    guidance = cfg.get("guidance_scale", 7.5)

    ids  = val_batch["input_ids"].to(device)
    mask = val_batch["attention_mask"].to(device)

    unwrap_enc  = accelerator.unwrap_model(text_enc)
    unwrap_unet = accelerator.unwrap_model(unet)

    with torch.no_grad():
        cond_hidden = unwrap_enc(input_ids=ids, attention_mask=mask).last_hidden_state
        B = ids.shape[0]
        if guidance > 1.0:
            uncond_ids  = torch.zeros_like(ids)
            uncond_mask = torch.ones_like(mask)
            uncond_hidden  = unwrap_enc(input_ids=uncond_ids, attention_mask=uncond_mask).last_hidden_state
            encoder_hidden = torch.cat([uncond_hidden, cond_hidden])
        else:
            encoder_hidden = cond_hidden

        val_scheduler.set_timesteps(30)
        latents = torch.randn(B, 4, 64, 64, device=device, dtype=cond_hidden.dtype)
        for t in val_scheduler.timesteps:
            if guidance > 1.0:
                pred = unwrap_unet(torch.cat([latents, latents]), t, encoder_hidden).sample
                noise_uncond, noise_cond = pred.chunk(2)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = unwrap_unet(latents, t, encoder_hidden).sample
            latents = val_scheduler.step(noise_pred, t, latents).prev_sample

        vae_dtype = next(vae.parameters()).dtype
        images = vae.decode((latents / vae.config.scaling_factor).to(vae_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)

    out_dir = Path(output_dir) / f"val_step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_t in enumerate(images):
        img_np = (img_t.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(out_dir / f"sample_{i:02d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=cfg.get("grad_accumulation", 1),
    )

    base_model  = cfg["base_model"]
    noise_sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    vae         = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet        = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    tokenizer   = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_enc    = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")

    vae.requires_grad_(False)
    text_enc.requires_grad_(False)

    lora_rank = cfg.get("unet_lora_rank", 16)
    lora_cfg  = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=cfg.get("lora_target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        lora_dropout=0.0, bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=cfg.get("lr_lora", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-2),
    )

    ec_json = str(PROJECT_DIR / "outputs" / cfg.get("ec_json", "ec_corpus_blip_ref_ep29.json"))
    with open(ec_json) as f:
        ec = json.load(f)
    train_nl = load_nl_captions(str(TRAIN_ANN))
    val_nl   = load_nl_captions(str(VAL_ANN))

    train_ds = NLCOCODataset(ec["train"], train_nl, tokenizer, cfg.get("image_size", 512))
    val_ds   = NLCOCODataset(ec["val"],   val_nl,   tokenizer, cfg.get("image_size", 512))

    batch_size   = cfg.get("batch_size", 16)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
    )

    rng = random.Random(cfg.get("seed", 42))
    val_indices  = rng.sample(range(len(val_ds)), min(8, len(val_ds)))
    val_batch_list = [val_ds[i] for i in val_indices]
    val_batch = {
        k: torch.stack([b[k] for b in val_batch_list])
        if isinstance(val_batch_list[0][k], torch.Tensor) else
        [b[k] for b in val_batch_list]
        for k in val_batch_list[0]
    }

    total_steps  = cfg.get("total_steps", 10000)
    warmup_steps = cfg.get("warmup_steps", 500)
    grad_accum   = cfg.get("grad_accumulation", 1)

    lr_sched = get_scheduler(
        "constant_with_warmup", optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    unet, text_enc, optimizer, train_loader, lr_sched = accelerator.prepare(
        unet, text_enc, optimizer, train_loader, lr_sched
    )

    num_update_steps = math.ceil(len(train_loader) / grad_accum)
    num_epochs = math.ceil(total_steps / num_update_steps)
    vae = vae.to(accelerator.device)
    vae = vae_to_precision(vae, cfg.get("mixed_precision", "fp16"))

    val_scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")
    ckpt_dir   = str(PROJECT_DIR / "checkpoints" / cfg.get("run_name", "pilot_NL"))
    output_dir = str(PROJECT_DIR / "outputs"     / cfg.get("run_name", "pilot_NL"))

    val_interval  = cfg.get("val_interval", 500)
    ckpt_interval = cfg.get("ckpt_interval", 2500)
    global_step   = 0

    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process)

    for epoch in range(num_epochs):
        unet.train()
        for batch in train_loader:
            if global_step >= total_steps:
                break
            with accelerator.accumulate(unet):
                images = batch["image"].to(accelerator.device)
                ids    = batch["input_ids"].to(accelerator.device)
                mask   = batch["attention_mask"].to(accelerator.device)

                latents = encode_images(vae, images.to(vae.dtype))
                noise   = torch.randn_like(latents)
                B       = latents.shape[0]
                ts      = torch.randint(0, noise_sched.config.num_train_timesteps,
                                        (B,), device=accelerator.device).long()
                noisy   = noise_sched.add_noise(latents, noise, ts)

                with torch.no_grad():
                    enc_hidden = text_enc(input_ids=ids, attention_mask=mask).last_hidden_state

                noise_pred = unet(noisy, ts, enc_hidden).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        cfg.get("max_grad_norm", 1.0),
                    )
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

                if global_step % val_interval == 0:
                    unet.eval()
                    validate_and_log(accelerator, unet, text_enc, vae, val_scheduler,
                                     val_batch, cfg, global_step, output_dir)
                    unet.train()

                if global_step % ckpt_interval == 0:
                    save_checkpoint(accelerator, unet, cfg, global_step, ckpt_dir)

            if global_step >= total_steps:
                break

    save_checkpoint(accelerator, unet, cfg, global_step, ckpt_dir)
    accelerator.end_training()
    progress.close()
    print(f"Training complete. Final step: {global_step}")


if __name__ == "__main__":
    main()
