"""Attention prior visualisation and export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from maa.attention.maa.block import MAAConfig, MicroscopyAwareAttention
from maa.attention.maa.boundary import BoundaryPrior
from maa.attention.maa.compactness import CompactnessPrior
from maa.attention.maa.density import DensityPrior
from maa.attention.maa.focus import FocusPrior
from maa.attention.maa.morphology import MorphologyPrior
from maa.attention.maa.organelle import OrganellePrior


def _to_numpy_map(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]
    x = x.numpy()
    x = x - x.min()
    denom = x.max() - x.min()
    if denom > 1e-8:
        x = x / denom
    return x


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend RGB uint8 image with a [0,1] heatmap using a simple jet-like map."""
    if image.dtype != np.uint8:
        img = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img = image
    h, w = img.shape[:2]
    heat_r = torch.from_numpy(heat).float().unsqueeze(0).unsqueeze(0)
    heat_r = F.interpolate(heat_r, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    # Approximate jet colormap without matplotlib dependency
    r = np.clip(1.5 - np.abs(4 * heat_r - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat_r - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat_r - 1), 0, 1)
    color = np.stack([r, g, b], axis=-1)
    blend = (1 - alpha) * (img.astype(np.float32) / 255.0) + alpha * color
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)


class PriorVisualizer:
    """Extract and save the six MAA prior maps for a batch of images."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.boundary = BoundaryPrior(learnable_edge=False).to(self.device)
        self.focus = FocusPrior().to(self.device)
        self.organelle = OrganellePrior().to(self.device)
        self.compactness = CompactnessPrior(use_foreground=True).to(self.device)
        self.morphology = MorphologyPrior().to(self.device)
        self.density = DensityPrior(use_mass=False).to(self.device)

    @torch.inference_mode()
    def prior_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(self.device)
        return {
            "boundary": self.boundary.edge_map(images),
            "focus": self.focus.sharpness_map(images),
            "organelle": self.organelle.activation_map(images),
            "morphology": self.morphology.ratio_map(images),
            "compactness_fg": (
                self.compactness.foreground(images)
                if self.compactness.foreground is not None
                else torch.ones_like(images[:, :1])
            ),
        }

    @torch.inference_mode()
    def bias_matrices(
        self,
        images: torch.Tensor,
        num_tokens: int = 256,
        tokens_h: int = 16,
        tokens_w: int = 16,
    ) -> dict[str, torch.Tensor]:
        images = images.to(self.device)
        return {
            "boundary": self.boundary(images, num_tokens, tokens_h, tokens_w),
            "focus": self.focus(images, num_tokens, tokens_h, tokens_w),
            "organelle": self.organelle(images, num_tokens, tokens_h, tokens_w),
            "compactness": self.compactness(images, num_tokens, tokens_h, tokens_w),
            "morphology": self.morphology(images, num_tokens, tokens_h, tokens_w),
            "density": self.density(images, num_tokens, tokens_h, tokens_w),
        }

    def save_overlays(
        self,
        images: torch.Tensor,
        out_dir: str | Path,
        *,
        prefix: str = "sample",
        max_items: int = 4,
    ) -> list[Path]:
        from PIL import Image

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        maps = self.prior_maps(images)
        paths: list[Path] = []
        n = min(images.size(0), max_items)
        for i in range(n):
            img = images[i].detach().cpu()
            if img.min() < 0:
                # Assume ImageNet-ish; denorm roughly to 0-1 for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
            img_np = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for name, tensor in maps.items():
                heat = _to_numpy_map(tensor[i : i + 1])
                overlay = overlay_heatmap(img_np, heat)
                path = out_dir / f"{prefix}_{i:03d}_{name}.png"
                Image.fromarray(overlay).save(path)
                paths.append(path)
        return paths


@torch.inference_mode()
def export_maa_attention(
    block: MicroscopyAwareAttention,
    tokens: torch.Tensor,
    images: torch.Tensor,
    out_path: str | Path,
) -> dict[str, Any]:
    """Run one MAA forward with return_attn and dump summary JSON + tensors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_out, extras = block(tokens, images, return_attn=True)
    attn = extras["attn"]
    bias = extras["bias"]
    parts = extras["bias_parts"]
    payload = {
        "attn_shape": list(attn.shape),
        "bias_shape": list(bias.shape),
        "lambda": {
            "boundary": float(block.lambda_boundary.detach().cpu()),
            "focus": float(block.lambda_focus.detach().cpu()),
            "organelle": float(block.lambda_organelle.detach().cpu()),
            "compactness": float(block.lambda_compactness.detach().cpu()),
            "morphology": float(block.lambda_morphology.detach().cpu()),
            "density": float(block.lambda_density.detach().cpu()),
        },
        "prior_enabled": dict(block.enable),
        "bias_part_means": {k: float(v.mean().cpu()) for k, v in parts.items()},
    }
    torch.save(
        {
            "attn": attn.cpu(),
            "bias": bias.cpu(),
            "parts": {k: v.cpu() for k, v in parts.items()},
            "tokens_out": tokens_out.cpu(),
        },
        out_path.with_suffix(".pt"),
    )
    out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def demo_priors(
    out_dir: str | Path = "runs/attention_viz",
    batch: int = 2,
    size: int = 256,
) -> None:
    """Write example prior overlays for random images with drawn shapes."""
    device = "cpu"
    images = torch.rand(batch, 3, size, size)
    # Draw discs / rods for visual structure
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    disc = ((xx - 0.2) ** 2 + (yy + 0.1) ** 2 < 0.08).float()
    rod = ((xx * 0.3 + yy).abs() < 0.05).float() * ((xx.abs() < 0.6).float())
    images[:, 0] = torch.clamp(images[:, 0] + 0.6 * disc + 0.4 * rod, 0, 1)
    viz = PriorVisualizer(device=device)
    paths = viz.save_overlays(images, out_dir, prefix="demo")
    bias = viz.bias_matrices(images, num_tokens=64, tokens_h=8, tokens_w=8)
    summary = {k: float(v.mean()) for k, v in bias.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "bias_means.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(paths)} overlays to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Visualise MAA priors")
    p.add_argument("--out", default="runs/attention_viz")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args(argv)
    demo_priors(args.out, args.batch, args.size)


if __name__ == "__main__":
    main()
