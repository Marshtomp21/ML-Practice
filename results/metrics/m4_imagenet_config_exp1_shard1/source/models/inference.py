"""Strict local checkpoint loading and counted, deterministic logit inference."""
from pathlib import Path

import numpy as np
import torch
from torchvision import models


def configure_runtime(seed=0, threads=4):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_chncxr(architecture, checkpoint, device="cuda"):
    """Load course two-class torchvision weights; no unsafe pickle fallback.

    Course checkpoint dictionaries also contain optimizer and NumPy scalar
    metadata. Allow only the specific scalar/dtype globals needed to read them.
    See https://docs.pytorch.org/docs/stable/notes/serialization.html.
    """
    constructors = {"resnet50": models.resnet50, "vgg13": models.vgg13}
    if architecture not in constructors:
        raise ValueError(f"Unsupported architecture: {architecture}")
    try:
        from numpy._core.multiarray import scalar
    except ImportError:
        from numpy.core.multiarray import scalar
    allowed = [np.dtype, type(np.dtype("float64")), type(np.dtype("float32"))]
    # Older torch releases lack tuple aliases; NumPy 1.x uses the original name.
    allowed.append((scalar, "numpy.core.multiarray.scalar")
                   if scalar.__module__ != "numpy.core.multiarray" else scalar)
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    model = constructors[architecture](weights=None, num_classes=2)
    model.load_state_dict(state, strict=True)
    return LogitPredictor(model, device)


def load_imagenet(architecture, checkpoint, device="cuda"):
    """Load a frozen torchvision ImageNet state dict from a local file."""
    constructors = {"resnet50": models.resnet50, "vgg13": models.vgg13}
    if architecture not in constructors:
        raise ValueError(f"Unsupported architecture: {architecture}")
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = constructors[architecture](weights=None, num_classes=1000)
    model.load_state_dict(state, strict=True)
    return LogitPredictor(model, device)


class LogitPredictor:
    def __init__(self, model, device="cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.samples = 0
        self.batches = 0

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __call__(self, batch):
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 4 or batch.shape[-1] != 3 or not np.isfinite(batch).all():
            raise ValueError("Expected finite NHWC RGB batch")
        tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
        with torch.inference_mode():
            logits = self.model(tensor.to(self.device)).cpu().numpy()
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("Model returned invalid logits")
        self.samples += len(batch)
        self.batches += 1
        return logits
