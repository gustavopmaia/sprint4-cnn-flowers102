import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import BATCH_SIZE, DATA, RAIZ, SPLITS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LADO_RESIZE, LADO = 256, 224  # resize 256 + crop 224, como no pdf da sprint 3


def gera_splits():
    # o split e o oficial do torchvision (train/val/test), mesmo da sprint 3
    from torchvision.datasets import Flowers102
    SPLITS.mkdir(exist_ok=True)
    for nome in ["train", "val", "test"]:
        ds = Flowers102(root=str(DATA), split=nome, download=True)
        df = pd.DataFrame({
            "path": [str(p.relative_to(RAIZ)) for p in ds._image_files],
            "label": ds._labels,
        })
        df.to_csv(SPLITS / f"{nome}.csv", index=False)


def le_split(nome):
    return pd.read_csv(SPLITS / f"{nome}.csv")


def _carrega(nome, lado_resize, crop_central):
    df = le_split(nome)
    cache = DATA / f"cache_{nome}_{lado_resize}.pt"
    if cache.exists():
        return torch.load(cache), torch.tensor(df.label.values)
    imgs = []
    for p in df.path:
        im = Image.open(RAIZ / p).convert("RGB").resize((lado_resize, lado_resize), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        if crop_central:
            o = (lado_resize - LADO) // 2
            x = x[:, o:o + LADO, o:o + LADO]
        imgs.append(x)
    x = torch.stack(imgs)
    torch.save(x, cache)
    return x, torch.tensor(df.label.values)


class Tensores(Dataset):
    def __init__(self, x, y, aug=None):
        self.x, self.y, self.aug = x, y, aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.aug is not None:
            x = self.aug(x)
        return x, self.y[i]


# augmentation so no treino
AUG = T.Compose([
    T.RandomCrop(LADO),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
])


def normaliza(x, device):
    x = x.to(device).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def loaders(seed):
    xt, yt = _carrega("train", LADO_RESIZE, crop_central=False)
    xv, yv = _carrega("val", LADO_RESIZE, crop_central=True)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(Tensores(xt, yt, AUG), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val = DataLoader(Tensores(xv, yv), batch_size=BATCH_SIZE, shuffle=False)
    return train, val


def loader_test():
    # so e chamado dentro do evaluate_test
    xs, ys = _carrega("test", LADO_RESIZE, crop_central=True)
    return DataLoader(Tensores(xs, ys), batch_size=BATCH_SIZE, shuffle=False), le_split("test")


def pesos_de_classe():
    # train e balanceado (10 por classe), entao os pesos sao todos 1
    cont = le_split("train").label.value_counts().sort_index()
    return (cont.sum() / (len(cont) * cont)).values
