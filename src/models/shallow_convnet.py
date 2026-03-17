import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        chans,
        classes,
        time_points,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length="auto",
        drop_prob=0.5,
    ):
        super().__init__()
        self.model = ShallowFBCSPNet(
            n_chans=chans,
            n_outputs=classes,
            n_times=time_points,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_filters_spat=n_filters_spat,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            final_conv_length=final_conv_length,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        # ShallowFBCSPNet expects (B, C, T) — 3D input
        # It does its own Ensure4d + Rearrange internally
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, C, T) → (B, C, T)
        return self.model(x)