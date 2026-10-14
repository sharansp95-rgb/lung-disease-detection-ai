"""
Reusable Grad-CAM implementation. Hooks into the last convolutional feature
block of a densenet121 backbone (model.backbone.features -- the output right
before the classifier's global-average-pool), so the same code works for any
checkpoint built on that architecture, just by passing in a different model
instance / class_idx.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Uses a forward hook that clones the target layer's output and calls
    retain_grad() on the clone, rather than register_full_backward_hook.
    densenet121's forward does F.relu(features, inplace=True) immediately
    after this layer -- an in-place op on a tensor a full-backward-hook is
    attached to is forbidden by autograd ("Output of BackwardHookFunction is
    a view and is being modified inplace"). Substituting a cloned tensor
    (via the forward hook's return value) sidesteps the conflict entirely:
    the in-place relu mutates the clone, and gradients w.r.t. that same
    clone are read back from its own .grad after backward()."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        out = out.clone()
        out.retain_grad()
        self.activations = out
        return out

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """input_tensor: [1, 3, H, W]. Returns (cam [Hf,Wf] in [0,1], class_idx used, full softmax probs)."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[0, class_idx]
        score.backward()

        gradients = self.activations.grad  # [1, C, H, W]
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-avg-pool of gradients
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)
        return cam, class_idx, probs[0].detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_handle.remove()


def get_target_layer(model):
    """Last conv-producing block before the classifier head."""
    if model.backbone_name != "densenet121":
        raise ValueError(
            f"Grad-CAM target-layer lookup only implemented for densenet121, got {model.backbone_name!r}"
        )
    return model.backbone.features


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """[3,H,W] normalized tensor -> uint8 HWC RGB numpy, for display/overlay."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay_cam(image_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = image_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap.astype(np.float32) * alpha + image_rgb_uint8.astype(np.float32) * (1 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
