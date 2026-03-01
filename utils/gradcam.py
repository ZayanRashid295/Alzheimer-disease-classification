"""
Grad-CAM for custom CNN: hook last conv output and gradients, compute heatmap, overlay on image.
"""
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM using target layer output and gradient w.r.t. that output.
    Works with any model that exposes a method returning the conv feature map
    (e.g. get_conv_output) and we run forward to get logits and backprop to get gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles = []

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self) -> None:
        self._hook_handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hook_handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run forward and backward, then compute Grad-CAM heatmap (H, W) as numpy.
        """
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.register_hooks()
        try:
            logits = self.model(x)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[0, target_class].backward()
            return self._heatmap()
        finally:
            self.remove_hooks()

    def _heatmap(self) -> np.ndarray:
        """Weights = mean of gradients over channels; then sum(weights * activations)."""
        a = self.activations
        g = self.gradients
        if a is None or g is None:
            raise RuntimeError("Activations or gradients not set. Run forward/backward first.")
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = (weights * a).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1).cpu().numpy()
        # resize to input spatial size and normalize
        cam = cam[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay heatmap on image. image: (H, W) or (H, W, 3), float [0,1] or uint8.
    Returns RGB numpy (H, W, 3) uint8. Uses matplotlib colormap (no OpenCV).
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    cmap = plt.get_cmap("jet")
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR,
        )
    ).astype(np.float32) / 255.0
    heatmap_rgb = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
    return overlay


def save_gradcam(
    heatmap: np.ndarray,
    image_tensor: torch.Tensor,
    save_path: Path,
    alpha: float = 0.5,
) -> None:
    """
    image_tensor: (1, C, H, W) or (C, H, W), normalized. Denormalize for visualization.
    """
    img = image_tensor.detach().cpu()
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = (img * 255).astype(np.uint8)
    overlay = overlay_heatmap(heatmap, img_uint8, alpha=alpha)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(save_path)
