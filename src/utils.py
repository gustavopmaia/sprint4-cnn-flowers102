import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("SPRINT4_RESULTS", RAIZ / "results"))
SPLITS = RAIZ / "splits"
DATA = RAIZ / "data"

# protocolo fixo, igual pra todos os experimentos
SEEDS = [42, 43, 44]
BATCH_SIZE = 32
LR = float(os.environ.get("SPRINT4_LR", 5e-4))
CLIP = float(os.environ.get("SPRINT4_CLIP", 0)) or None
MAX_EPOCAS = int(os.environ.get("SPRINT4_EPOCAS", 50))
PACIENCIA = 10
NUM_CLASSES = 102


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def versoes():
    import sklearn, torchvision
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": str(get_device()),
    }


def salva_json(obj, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def le_json(caminho):
    return json.loads(Path(caminho).read_text())


# mesmo formato nos slides e na validacao
def f3(x):
    return f"{x:.3f}".replace(".", ",")


def fpar(n):
    return f"{int(n):,}".replace(",", ".")
