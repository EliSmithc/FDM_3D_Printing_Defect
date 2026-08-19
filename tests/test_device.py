"""Tests for runtime device selection."""

from __future__ import annotations

import pytest
import torch

from fdm_defect.device import describe_device, resolve_device


class TestResolveDevice:
    def test_returns_an_available_device_by_default(self):
        assert resolve_device().type in {"cuda", "mps", "cpu"}

    def test_cpu_can_always_be_requested(self):
        assert resolve_device("cpu").type == "cpu"

    def test_unavailable_backend_fails_loudly(self, monkeypatch):
        """Silently falling back to cpu would make a run mysteriously slow."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")

    def test_prefers_cuda_when_present(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device().type == "cuda"

    def test_falls_back_to_cpu_when_nothing_is_accelerated(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device().type == "cpu"


class TestDescribeDevice:
    def test_names_the_backend(self):
        assert "cpu" in describe_device(torch.device("cpu"))
