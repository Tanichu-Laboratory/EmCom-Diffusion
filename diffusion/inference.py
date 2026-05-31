"""
Inference: generate images from emergent-language captions and build HTML gallery.

For each of the 3 settings (A, B, C) generate images for a fixed val-100 subset,
then produce outputs/qualitative.html.

Usage:
  python -m diffusion.inference \
    --ckpts checkpoints/pilot_A/step_010000 \
            checkpoints/pilot_B/step_010000 \
            checkpoints/pilot_C/step_010000 \
    --labels A B C \
    --n_samples 100
"""

import argparse
import base64
import json
import os
import random
import sys
from io import BytesIO
from pathlib import Path

import torch
import yaml
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from ec_text_encoder import build_ec_text_encoder
from data import build_datasets, ECCOCODataset

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(_HERE.parent)))
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "data/coco2017"))
TRAIN_ANN = COCO_ROOT / "annotations" / "captions_train2017.json"
VAL_ANN   = COCO_ROOT / "annotations" / "captions_val2017.json"


def _pad_to_77(ec_tokens, attention_mask, pad_id):
    B, T = ec_tokens.shape
    if T >= 77:
        return ec_tokens[:, :77], attention_mask[:, :77]
    pad = 77 - T
    ids  = torch.full((B, pad), pad_id, dtype=torch.long, device=ec_tokens.device)
    mask = torch.zeros((B, pad), dtype=torch.long, device=ec_tokens.device)
    return torch.cat([ec_tokens, ids], 1), torch.cat([attention_mask, mask], 1)


def load_checkpoint(ckpt_dir, base_model, device):
    """Load text_encoder + unet from a training checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "train_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    text_enc = build_ec_text_encoder(
        vocab_size=cfg["vocab_size"],
        pad_token_id=cfg["pad_token_id"],
        unfreeze_top_k_layers=cfg.get("unfreeze_top_k_layers", 0),
    )
    state = torch.load(str(ckpt_dir / "text_encoder_state.pt"), map_location="cpu")
    text_enc.load_state_dict(state)
    text_enc = text_enc.to(device).eval()

    lora_dir = ckpt_dir / "unet_lora"
    if lora_dir.exists():
        from peft import PeftModel
        unet_base = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        unet = PeftModel.from_pretrained(unet_base, str(lora_dir))
    else:
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet = unet.to(device).eval()

    return text_enc, unet, cfg


@torch.no_grad()
def generate_images(text_enc, unet, vae, scheduler, ec_tokens, attn_mask,
                    pad_id, n_steps=50, guidance=7.5, device="cuda"):
    """Generate images with classifier-free guidance."""
    ids, mask = _pad_to_77(ec_tokens, attn_mask, pad_id)
    cond_hidden = text_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    B = ids.shape[0]
    if guidance > 1.0:
        uncond_ids  = torch.full((B, 77), pad_id, dtype=torch.long, device=device)
        uncond_mask = torch.ones(B, 77, dtype=torch.long, device=device)
        uncond_hidden  = text_enc(input_ids=uncond_ids, attention_mask=uncond_mask).last_hidden_state
        encoder_hidden = torch.cat([uncond_hidden, cond_hidden])  # [2B, 77, 768]
    else:
        encoder_hidden = cond_hidden

    latents = torch.randn(B, 4, 64, 64, device=device, dtype=cond_hidden.dtype)
    scheduler.set_timesteps(n_steps)

    for t in scheduler.timesteps:
        if guidance > 1.0:
            noise_pred_raw = unet(torch.cat([latents, latents]), t, encoder_hidden).sample
            noise_uncond, noise_cond = noise_pred_raw.chunk(2)
            noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(latents, t, encoder_hidden).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    images = vae.decode(latents / vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images  # [B, 3, H, W] float32


def tensor_to_pil(img_t):
    arr = (img_t.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def img_to_b64(img: Image.Image, size: int = 160) -> str:
    img = img.copy()
    img.thumbnail((size, size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def path_to_b64(path: str, size: int = 160) -> str:
    return img_to_b64(Image.open(path).convert("RGB"), size)


@torch.no_grad()
def generate_nl_images(captions, base_model, vae, scheduler,
                       n_steps=50, guidance=7.5, batch_size=8, device="cuda",
                       nl_ft_ckpt=None):
    """Generate images from NL captions.

    nl_ft_ckpt: path to a pilot_NL checkpoint dir (unet_lora/ subdir expected).
                When None, uses the base SD UNet (unfinetuned).
    """
    tokenizer  = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_enc   = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device).eval()
    if nl_ft_ckpt is not None:
        from peft import PeftModel
        unet_base = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        unet = PeftModel.from_pretrained(unet_base, str(Path(nl_ft_ckpt) / "unet_lora")).to(device).eval()
    else:
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device).eval()

    imgs_out = []
    for start in tqdm(range(0, len(captions), batch_size), desc="NL-SD"):
        batch_caps = captions[start : start + batch_size]
        bsz = len(batch_caps)

        enc = tokenizer(batch_caps, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(device)
        cond_hidden = text_enc(**enc).last_hidden_state

        if guidance > 1.0:
            uncond_enc = tokenizer([""] * bsz, padding="max_length", max_length=77,
                                   return_tensors="pt").to(device)
            uncond_hidden = text_enc(**uncond_enc).last_hidden_state
            encoder_hidden = torch.cat([uncond_hidden, cond_hidden])
        else:
            encoder_hidden = cond_hidden

        latents = torch.randn(bsz, 4, 64, 64, device=device, dtype=cond_hidden.dtype)
        scheduler.set_timesteps(n_steps)

        for t in scheduler.timesteps:
            if guidance > 1.0:
                noise_pred_raw = unet(torch.cat([latents, latents]), t, encoder_hidden).sample
                noise_uncond, noise_cond = noise_pred_raw.chunk(2)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(latents, t, encoder_hidden).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = vae.decode(latents / vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        for img_t in images:
            imgs_out.append(tensor_to_pil(img_t))

    del text_enc, unet
    torch.cuda.empty_cache()
    return imgs_out


def build_html(val_entries, generated, out_path, labels, nl_caps, nl_images=None):
    """
    val_entries: list of {"image_id", "image_path", "ec_tokens"}
    generated:   {label: [PIL.Image, ...]} length = len(val_entries) per label
    nl_caps:     {image_id: [str, ...]}
    nl_images:   [PIL.Image, ...] or None — base SD + NL caption baseline
    """
    rows = []
    for i, entry in enumerate(val_entries):
        orig_b64 = path_to_b64(entry["image_path"])
        row = f'<td><img src="data:image/jpeg;base64,{orig_b64}" width="150"><br>'
        row += f'<small>id={entry["image_id"]}</small></td>'

        for label in labels:
            gen_b64 = img_to_b64(generated[label][i])
            row += f'<td><img src="data:image/jpeg;base64,{gen_b64}" width="150"><br>'
            row += f'<small>{label}</small></td>'

        if nl_images is not None:
            nl_b64 = img_to_b64(nl_images[i])
            row += f'<td><img src="data:image/jpeg;base64,{nl_b64}" width="150"><br>'
            row += f'<small>SD-NL</small></td>'

        nl = nl_caps.get(entry["image_id"], [""])[0]
        tok = entry["ec_tokens"]
        row += f'<td style="font-size:11px">{nl}</td>'
        row += f'<td style="font-size:11px">{tok}</td>'
        rows.append(f"<tr>{row}</tr>")

    header_labels = " ".join(f"<th>Gen ({l})</th>" for l in labels)
    nl_header = "<th>Gen (SD-NL)</th>" if nl_images is not None else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>EC2Img Qualitative Results</title>
<style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border: 1px solid #ccc; padding: 4px; vertical-align: top; }}
  th {{ background: #333; color: #fff; }}
</style></head><body>
<h1>EC→Image Generation: Qualitative Comparison</h1>
<p>Settings: {", ".join(labels)}{" + SD-NL baseline" if nl_images is not None else ""}</p>
<table>
<tr><th>Original</th>{header_labels}{nl_header}<th>NL Caption</th><th>EC Tokens</th></tr>
{"".join(rows)}
</table>
</body></html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved HTML: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts",    nargs="+", required=True,
                        help="Checkpoint dirs for each setting")
    parser.add_argument("--labels",   nargs="+", default=None,
                        help="Label for each checkpoint (default: A B C ...)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--n_steps",   type=int, default=50)
    parser.add_argument("--guidance",  type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ec_json",
                        default=str(PROJECT_DIR / "outputs" / "ec_captions.json"))
    parser.add_argument("--out_html",
                        default=str(PROJECT_DIR / "outputs" / "qualitative.html"))
    parser.add_argument("--out_img_dir",
                        default=str(PROJECT_DIR / "outputs" / "generated_images"))
    parser.add_argument("--input_type", choices=["ec", "random", "fixed"], default="ec",
                        help="ec: real EC tokens; random: random tokens per sample; "
                             "fixed: same token sequence for all samples")
    parser.add_argument("--nl_baseline", action="store_true",
                        help="Also generate SD-NL baseline column in HTML (ec mode only)")
    parser.add_argument("--nl_only", action="store_true",
                        help="Skip EC generation; load existing images and add NL baseline to HTML")
    parser.add_argument("--nl_ft_ckpt", default=None,
                        help="Path to pilot_NL checkpoint dir; when set, uses COCO-finetuned UNet "
                             "for NL baseline (fair comparison). Images saved to NL_ft/ subdir.")
    args = parser.parse_args()

    labels = args.labels or [chr(65 + i) for i in range(len(args.ckpts))]
    if not args.nl_only:
        assert len(labels) == len(args.ckpts)

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load val entries
    with open(args.ec_json) as f:
        ec = json.load(f)
    val_all = ec["val"]
    val_entries = random.sample(val_all, min(args.n_samples, len(val_all)))

    from data import load_nl_captions
    nl_caps = load_nl_captions(str(VAL_ANN))

    # determine base_model from first ckpt config
    first_cfg_path = Path(args.ckpts[0]) / "train_config.yaml"
    with open(first_cfg_path) as f:
        first_cfg = yaml.safe_load(f)
    base_model = first_cfg["base_model"]
    pad_id     = first_cfg["pad_token_id"]

    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # --nl_only: load existing EC images and just generate NL baseline
    if args.nl_only:
        generated = {}
        for label in labels:
            img_dir = Path(args.out_img_dir) / label
            paths = sorted(img_dir.glob("*.png"))[:len(val_entries)]
            generated[label] = [Image.open(p).convert("RGB") for p in paths]
        captions = [nl_caps.get(e["image_id"], [""])[0] for e in val_entries]

        nl_subdir = "NL_ft" if args.nl_ft_ckpt else "NL"
        tag = f"COCO-finetuned NL (pilot_NL)" if args.nl_ft_ckpt else "base SD, no finetuning"
        print(f"\n=== Generating NL baseline ({tag}) ===")
        nl_out_dir = Path(args.out_img_dir) / nl_subdir
        nl_out_dir.mkdir(parents=True, exist_ok=True)
        nl_images = generate_nl_images(
            captions, base_model, vae, scheduler,
            n_steps=args.n_steps, guidance=args.guidance,
            batch_size=args.batch_size, device=str(device),
            nl_ft_ckpt=args.nl_ft_ckpt,
        )
        for idx, pil in enumerate(nl_images):
            pil.save(str(nl_out_dir / f"{idx:04d}_imgid{val_entries[idx]['image_id']}.png"))
        build_html(val_entries, generated, args.out_html, labels, nl_caps, nl_images=nl_images)
        print("Done.")
        return

    generated = {}

    input_type   = args.input_type
    seq_len      = len(val_entries[0]["ec_tokens"])
    fixed_tokens = val_entries[0]["ec_tokens"]

    for label, ckpt_dir in zip(labels, args.ckpts):
        print(f"\n=== Generating with setting {label} ({input_type}) from {ckpt_dir} ===")
        text_enc, unet, cfg = load_checkpoint(ckpt_dir, base_model, device)
        vocab_size = cfg["vocab_size"]   # includes pad token (pad_id = vocab_size - 1)

        imgs_out = []
        dir_suffix = "" if input_type == "ec" else f"_{input_type}"
        out_dir = Path(args.out_img_dir) / (label + dir_suffix)
        out_dir.mkdir(parents=True, exist_ok=True)

        _rng = random.Random(args.seed)

        for start in tqdm(range(0, len(val_entries), args.batch_size), desc=label):
            batch_e = val_entries[start : start + args.batch_size]
            bsz = len(batch_e)

            if input_type == "ec":
                tok_list = [e["ec_tokens"] for e in batch_e]
            elif input_type == "random":
                tok_list = [
                    [_rng.randint(0, vocab_size - 2) for _ in range(seq_len)]
                    for _ in range(bsz)
                ]
            else:  # fixed
                tok_list = [fixed_tokens] * bsz

            max_len = max(len(t) for t in tok_list)
            ids_t   = torch.full((bsz, max_len), pad_id, dtype=torch.long)
            mask_t  = torch.zeros(bsz, max_len, dtype=torch.long)
            for b, toks in enumerate(tok_list):
                ids_t[b, :len(toks)]  = torch.tensor(toks)
                mask_t[b, :len(toks)] = 1
            ids_t  = ids_t.to(device)
            mask_t = mask_t.to(device)

            imgs_t = generate_images(
                text_enc, unet, vae, scheduler,
                ids_t, mask_t, pad_id,
                n_steps=args.n_steps, guidance=args.guidance, device=str(device),
            )
            for i, img_t in enumerate(imgs_t):
                pil = tensor_to_pil(img_t)
                global_idx = start + i
                pil.save(str(out_dir / f"{global_idx:04d}_imgid{val_entries[global_idx]['image_id']}.png"))
                imgs_out.append(pil)

        generated[label] = imgs_out
        del text_enc, unet
        torch.cuda.empty_cache()

    if input_type == "ec":
        nl_images = None
        if args.nl_baseline:
            print("\n=== Generating NL baseline (base SD, no finetuning) ===")
            captions = [nl_caps.get(e["image_id"], [""])[0] for e in val_entries]
            nl_out_dir = Path(args.out_img_dir) / "NL"
            nl_out_dir.mkdir(parents=True, exist_ok=True)
            nl_images = generate_nl_images(
                captions, base_model, vae, scheduler,
                n_steps=args.n_steps, guidance=args.guidance,
                batch_size=args.batch_size, device=str(device),
            )
            for idx, pil in enumerate(nl_images):
                pil.save(str(nl_out_dir / f"{idx:04d}_imgid{val_entries[idx]['image_id']}.png"))
        build_html(val_entries, generated, args.out_html, labels, nl_caps, nl_images=nl_images)
    print("Done.")


if __name__ == "__main__":
    main()
