import hashlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.data import le_split  # noqa: E402

ok = True


def checa(desc, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + desc)


sp = {n: le_split(n) for n in ["train", "val", "test"]}

# numeros do pdf da sprint 3
checa("train tem 1020 imagens", len(sp["train"]) == 1020)
checa("val tem 1020 imagens", len(sp["val"]) == 1020)
checa("test tem 6149 imagens", len(sp["test"]) == 6149)
checa("total 8189", sum(len(d) for d in sp.values()) == 8189)
checa("102 classes", all(d.label.nunique() == 102 for d in sp.values()))
checa("train: 10 imagens por classe", (sp["train"].label.value_counts() == 10).all())
checa("val: 10 imagens por classe", (sp["val"].label.value_counts() == 10).all())
t = sp["test"].label.value_counts()
print(f"info: test por classe vai de {t.min()} a {t.max()}")

# os csv batem com o split oficial que o torchvision entrega (o que a sprint 3 usou)
from torchvision.datasets import Flowers102  # noqa: E402
for n in sp:
    ds = Flowers102(root=str(RAIZ / "data"), split=n, download=False)
    mesmos = [str(p.relative_to(RAIZ)) for p in ds._image_files] == list(sp[n].path)
    checa(f"{n}.csv igual ao Flowers102(split='{n}') da sprint 3 (caminhos e rotulos)",
          mesmos and list(ds._labels) == list(sp[n].label))

# vazamento por caminho
nomes = list(sp)
for i in range(3):
    for j in range(i + 1, 3):
        a, b = nomes[i], nomes[j]
        checa(f"sem interseccao de caminhos entre {a} e {b}", not set(sp[a].path) & set(sp[b].path))


def sha(p):
    return hashlib.sha256((RAIZ / p).read_bytes()).hexdigest()


# o Flowers102 original traz 2 pares de arquivos identicos entre splits (train/test e val/test).
# Nao da pra mexer porque o split tem que ser o da sprint 3, entao so aceitamos esses 2 pares.
CONHECIDOS = {
    ("train", "test"): {("image_07307.jpg", "image_07309.jpg")},
    ("val", "test"): {("image_08077.jpg", "image_08067.jpg")},
    ("train", "val"): set(),
}
hs = {n: sp[n].assign(h=sp[n].path.map(sha)) for n in sp}
for (a, b), esperados in CONHECIDOS.items():
    m = hs[a].merge(hs[b], on="h")
    achados = {(Path(x).name, Path(y).name) for x, y in zip(m.path_x, m.path_y)}
    checa(f"hash do arquivo entre {a} e {b}: {len(achados)} duplicata(s), so as conhecidas do dataset original", achados == esperados)

sys.exit(0 if ok else 1)
