"""Shared utilities for train.py and train_nl.py."""

from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML config, merging base_config if specified."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("base_config", None)
    if base_name:
        base_path = config_path.parent / base_name
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def encode_images(vae, images):
    with torch.no_grad():
        return vae.encode(images).latent_dist.sample() * vae.config.scaling_factor


def vae_to_precision(vae, mixed_precision: str):
    """Cast VAE to match the training mixed precision setting."""
    if mixed_precision == "fp16":
        vae = vae.to(torch.float16)
    elif mixed_precision == "bf16":
        vae = vae.to(torch.bfloat16)
    return vae
