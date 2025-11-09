# Ultralytics \U0001F680 AGPL-3.0 License - https://ultralytics.com/license
"""Swin Transformer backbone module."""

from __future__ import annotations

import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The SwinT backbone requires the 'timm' package. Install it with 'pip install timm'.") from exc


class SwinT(nn.Module):
    """Wrapper around the Swin-Tiny backbone from timm that returns multi-scale feature maps."""

    def __init__(
        self,
        pretrained: bool = True,
        in_chans: int = 3,
        out_indices: tuple[int, int, int] = (1, 2, 3),
        drop_path_rate: float | None = None,
        **kwargs,
    ) -> None:
        """Initialise the Swin-Tiny backbone."""
        super().__init__()

        swin_kwargs = dict(features_only=True, in_chans=in_chans, out_indices=out_indices)
        if drop_path_rate is not None:
            swin_kwargs["drop_path_rate"] = drop_path_rate
        swin_kwargs.update(kwargs)

        self.out_indices = tuple(out_indices)
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            **swin_kwargs,
        )
        if hasattr(self.model, "feature_info"):
            channels = self.model.feature_info.channels()
            self.channels = list(channels) if channels is not None else None
        else:  # pragma: no cover
            self.channels = None

    def forward(self, x):
        """Return the feature maps for the requested indices."""
        features = self.model(x)
        return list(features)
