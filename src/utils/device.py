import torch

def get_device():

    if torch.cuda.is_available():
        print("GPU detectada.")
        return torch.device("cuda")
    else:
        print("GPU não encontrada. Usando CPU.")
        return torch.device("cpu")
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")