import os
from dotenv import load_dotenv
import torch

# =====================================================
# Carrega variáveis do .env
# =====================================================

load_dotenv()

# =====================================================
# Device
# =====================================================

DEVICE = get_device()

# =====================================================
# Treinamento
# =====================================================

EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 2

# =====================================================
# Banco
# =====================================================

DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}