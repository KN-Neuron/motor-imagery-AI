"""Tests for the modular EEGNet pipeline."""

import numpy as np
import pytest
import torch

from src.models.eegnet import EEGNet
from src.data.dataset import EEGDataset
from src.config import load_config


class TestEEGNet:
    """Test the EEGNet model."""

    def test_forward_3class_64ch(self):
        model = EEGNet(chans=64, classes=3, time_points=641)
        x = torch.randn(4, 64, 641)
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_2class_21ch(self):
        model = EEGNet(chans=21, classes=2, time_points=641)
        x = torch.randn(4, 21, 641)
        out = model(x)
        assert out.shape == (4, 2)

    def test_forward_with_4d_input(self):
        model = EEGNet(chans=21, classes=2, time_points=641)
        x = torch.randn(4, 1, 21, 641)
        out = model(x)
        assert out.shape == (4, 2)

    def test_different_time_points(self):
        for tp in [161, 321, 481, 641]:
            model = EEGNet(chans=21, classes=2, time_points=tp)
            x = torch.randn(2, 21, tp)
            out = model(x)
            assert out.shape == (2, 2)

    def test_custom_hyperparams(self):
        model = EEGNet(
            chans=21, classes=3, time_points=641,
            f1=16, f2=32, d=2, dropout_rate=0.25,
        )
        x = torch.randn(2, 21, 641)
        out = model(x)
        assert out.shape == (2, 3)

    def test_gradient_flow(self):
        model = EEGNet(chans=21, classes=2, time_points=641)
        x = torch.randn(4, 21, 641)
        y = torch.randint(0, 2, (4,))
        loss_fn = torch.nn.CrossEntropyLoss()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestEEGDataset:
    """Test the EEGDataset."""

    def test_basic(self):
        X = np.random.randn(10, 21, 641).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        ds = EEGDataset(X, y)
        assert len(ds) == 10
        x_item, y_item = ds[0]
        assert x_item.shape == (21, 641)
        assert y_item.dtype == torch.long

    def test_augmentation(self):
        X = np.zeros((5, 21, 641), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)
        ds = EEGDataset(X, y, augment=True)
        x1, _ = ds[0]
        # Augmented should not be all zeros
        assert not torch.allclose(x1, torch.zeros_like(x1))


class TestConfig:
    """Test config loading."""

    def test_load_default(self):
        cfg = load_config()
        assert cfg["seed"] == 42
        assert cfg["eegnet"]["f2"] == 16
        assert len(cfg["channels"]["motor_channels"]) == 21
        assert cfg["training"]["scheduler_T_max"] == 50

    def test_override(self):
        cfg = load_config(overrides={"seed": 123, "training": {"epochs": 10}})
        assert cfg["seed"] == 123
        assert cfg["training"]["epochs"] == 10
        # Other fields preserved
        assert cfg["training"]["lr"] == 0.001
