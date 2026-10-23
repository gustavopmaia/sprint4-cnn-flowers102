import sys

from .evaluate import agrega
from .models import EXPERIMENTOS
from .train import treina
from .utils import SEEDS

if __name__ == "__main__":
    nomes = sys.argv[1:] or list(EXPERIMENTOS)
    for nome in nomes:
        for s in SEEDS:
            treina(nome, s)
        agrega()
