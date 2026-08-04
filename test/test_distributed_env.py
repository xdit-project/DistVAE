"""Which distributed backend DistVAE asks for, on machines with and without an accelerator."""

import torch

from distvae.utils import DistributedEnv


def test_cpu_only_machines_get_gloo(monkeypatch):
    # Sharding is correctness-testable on CPU, so a machine without an accelerator has to be
    # offered a backend rather than refused one.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "musa"):
        monkeypatch.setattr(torch.musa, "is_available", lambda: False)
    assert DistributedEnv.get_torch_distributed_backend() == "gloo"


def test_cuda_is_still_preferred_where_it_exists(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DistributedEnv.get_torch_distributed_backend() == "nccl"


def test_the_device_type_agrees_with_the_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "musa"):
        monkeypatch.setattr(torch.musa, "is_available", lambda: False)
    assert DistributedEnv.get_device_type() == "cpu"
    assert DistributedEnv.get_device() == torch.device("cpu")
