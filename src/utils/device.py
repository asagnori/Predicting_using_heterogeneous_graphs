"""Runtime device detection and validation."""

from __future__ import annotations

import torch


def _can_allocate(device: torch.device) -> bool:
    """Check that the selected backend can allocate and use a tensor."""
    try:
        tensor = torch.zeros(1, device=device)
        _ = tensor + 1
    except RuntimeError:
        return False
    return True


def get_device() -> torch.device:
    """Select and validate CUDA, MPS, or CPU in that priority order."""
    candidates: list[tuple[str, bool]] = [
        ("cuda", torch.cuda.is_available()),
        (
            "mps",
            torch.backends.mps.is_built()
            and torch.backends.mps.is_available(),
        ),
        ("cpu", True),
    ]

    for name, available in candidates:
        device = torch.device(name)
        if available and _can_allocate(device):
            print(f"Dispositivo selecionado: {device}.")
            return device
        if available:
            print(f"Dispositivo {device} detectado, mas falhou na validação.")

    raise RuntimeError(
        "Nenhum dispositivo PyTorch disponível para execução, "
        "incluindo CPU."
    )
