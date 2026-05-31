"""
Train the Referential Game (game='referential').

Usage (single GPU):
    python -m ec_game.train --config ec_game/configs/config_referential.yaml

Usage (multi-GPU, torchrun):
    torchrun --nproc_per_node=4 -m ec_game.train \
        --config ec_game/configs/config_referential.yaml
"""

import argparse
import datetime
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

_emcom_spec = importlib.util.spec_from_file_location(
    'simple_emcom_models', str(_HERE / 'models' / 'emcom.py')
)
_emcom_mod = importlib.util.module_from_spec(_emcom_spec)
_emcom_spec.loader.exec_module(_emcom_mod)
build_model = _emcom_mod.build_model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ImageListDataset(Dataset):
    """Loads images from a JSON file that contains a list of {"image": path} dicts."""

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    def __init__(self, ann_file: str):
        with open(ann_file) as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path = self.entries[idx]['image']
        img = Image.open(path).convert('RGB')
        return self._TRANSFORM(img), path


# ──────────────────────────────────────────────────────────────────────────────
# LR / temperature schedules
# ──────────────────────────────────────────────────────────────────────────────

def get_tau(step, tau_init, tau_min, anneal_steps):
    frac = min(step / max(anneal_steps, 1), 1.0)
    return tau_init - frac * (tau_init - tau_min)


def get_lr(step, warmup_steps, init_lr, warmup_lr, min_lr, total_steps):
    if step < warmup_steps:
        return warmup_lr + (init_lr - warmup_lr) * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, config,
                    global_step, writer=None):
    model.train()
    tau_init     = config.get('tau_init', 1.0)
    tau_min      = config.get('tau_min', 0.5)
    anneal_steps = config.get('tau_anneal_steps', 30000)
    w_ref        = config.get('loss_weight_ref', 1.0)
    w_recon      = config.get('loss_weight_recon', 0.0)
    warmup_steps = config.get('warmup_steps', 1000)
    init_lr      = config.get('init_lr', 3e-4)
    warmup_lr    = config.get('warmup_lr', 1e-6)
    min_lr       = config.get('min_lr', 1e-6)
    total_steps  = config.get('_total_steps', 30000)
    lambda_ent   = config.get('lambda_ent', 0.0)

    metric_sums = {'loss': 0, 'loss_ref': 0, 'loss_recon': 0, 'ent': 0}
    t0 = time.time()

    for step, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)

        lr  = get_lr(global_step, warmup_steps, init_lr, warmup_lr, min_lr, total_steps)
        tau = get_tau(global_step, tau_init, tau_min, anneal_steps)
        set_lr(optimizer, lr)

        loss_ref, loss_recon, tokens, token_ent = model(imgs, tau=tau)
        loss = w_ref * loss_ref + w_recon * loss_recon
        if lambda_ent > 0.0:
            loss = loss - lambda_ent * token_ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        metric_sums['loss']       += loss.item()
        metric_sums['loss_ref']   += loss_ref.item()
        metric_sums['loss_recon'] += loss_recon.item()
        metric_sums['ent']        += token_ent.item()

        if step % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (len(loader) - step - 1)
            print(
                f"Train Epoch [{epoch}] [{step:4d}/{len(loader)}]"
                f"  lr={lr:.2e}  tau={tau:.3f}"
                f"  loss={loss.item():.4f}"
                f"  ref={loss_ref.item():.4f}"
                f"  ent={token_ent.item():.3f}"
                f"  eta={datetime.timedelta(seconds=int(eta))}"
            )

        if writer is not None and global_step % 50 == 0:
            writer.add_scalar('train/loss',     loss.item(),       global_step)
            writer.add_scalar('train/loss_ref', loss_ref.item(),   global_step)
            writer.add_scalar('train/ent',      token_ent.item(),  global_step)
            writer.add_scalar('train/lr',       lr,                global_step)
            writer.add_scalar('train/tau',      tau,               global_step)

    n = len(loader)
    return {k: v / n for k, v in metric_sums.items()}, global_step


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    yaml = YAML(typ='rt')
    with open(args.config) as f:
        config = dict(yaml.load(f))

    config.setdefault('game', 'referential')
    config.setdefault('loss_weight_ref',   1.0)
    config.setdefault('loss_weight_recon', 0.0)

    # ── Distributed setup ──
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        device  = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        rank = 0; world_size = 1; is_main = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Output dir ──
    output_dir = Path(args.output_dir) / f"{config['game']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Output: {output_dir}")

    # ── TensorBoard ──
    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        except ImportError:
            pass

    # ── Dataset ──
    img_list = args.img_list or config.get('train_file', [])
    if isinstance(img_list, str):
        img_list = [img_list]
    if not img_list:
        raise ValueError("Provide --img_list or set train_file in the config.")

    dataset = ImageListDataset(img_list[0])

    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=config.get('batch_size', 128),
            sampler=sampler, num_workers=config.get('num_workers', 8),
            pin_memory=True, drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=config.get('batch_size', 128),
            shuffle=True, num_workers=config.get('num_workers', 8),
            pin_memory=True, drop_last=True,
        )

    steps_per_epoch = len(loader)
    max_epoch       = config.get('max_epoch', 30)
    config['_total_steps'] = steps_per_epoch * max_epoch

    # ── Model ──
    model = build_model(config).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('warmup_lr', 1e-6),
        weight_decay=config.get('weight_decay', 0.05),
    )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params/1e6:.1f}M")
        print(f"Steps/epoch: {steps_per_epoch}  Total epochs: {max_epoch}")

    global_step = 0
    save_interval = config.get('save_checkpoint_interval', 5)

    for epoch in range(max_epoch):
        if world_size > 1:
            sampler.set_epoch(epoch)

        train_metrics, global_step = train_one_epoch(
            model, loader, optimizer, device, epoch, config,
            global_step, writer=writer if is_main else None,
        )

        if is_main:
            print(f"\n[Epoch {epoch}] "
                  f"loss={train_metrics['loss']:.4f}  "
                  f"ref={train_metrics['loss_ref']:.4f}")

            if (epoch + 1) % save_interval == 0 or epoch == max_epoch - 1:
                ckpt_path = output_dir / f'checkpoint_{epoch:02d}.pth'
                model_state = (model.module.state_dict()
                               if hasattr(model, 'module') else model.state_dict())
                torch.save({'model': model_state, 'epoch': epoch, 'config': config},
                           ckpt_path)
                print(f"  Saved: {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--img_list', default=None,
                        help='Path to image-list JSON (overrides config train_file)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where checkpoints are saved')
    args = parser.parse_args()
    main(args)
