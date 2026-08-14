import torch

def get_device():

    if torch.cuda.is_available():
        print("GPU detectada.")
        return torch.device("cuda")
    else:
        print("GPU não encontrada. Usando CPU.")
        return torch.device("cpu")