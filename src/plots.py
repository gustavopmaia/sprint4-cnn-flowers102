import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import DATA, LADO, LADO_RESIZE, le_split
from .models import EXPERIMENTOS
from .utils import RESULTS, SEEDS, le_json

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})
COR = "#1f4e79"
COR2 = "#c0504d"


def hist_medio(nome):
    hs = [le_json(RESULTS / "runs" / f"{nome}_s{s}.json")["hist"] for s in SEEDS
          if (RESULTS / "runs" / f"{nome}_s{s}.json").exists()]
    n = max(len(h["val_loss"]) for h in hs)
    out = {}
    for k in hs[0]:
        m = np.full((len(hs), n), np.nan)
        for i, h in enumerate(hs):
            m[i, :len(h[k])] = h[k]
        out[k] = np.nanmean(m, axis=0)
    return out


def curvas_exp(nome, ax=None):
    # curva media entre seeds (rodadas que pararam antes entram so ate onde foram)
    h = hist_medio(nome)
    ep = np.arange(1, len(h["val_loss"]) + 1)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].plot(ep, h["train_loss"], color=COR, label="treino")
    ax[0].plot(ep, h["val_loss"], color=COR2, label="validação")
    ax[0].set(xlabel="época", ylabel="loss", title=f"{nome}: loss")
    ax[1].plot(ep, h["train_acc"], color=COR, label="treino")
    ax[1].plot(ep, h["val_acc"], color=COR2, label="validação")
    ax[1].set(xlabel="época", ylabel="acurácia", title=f"{nome}: acurácia")
    ax[0].legend()
    return ax


def todas_curvas(salvar=True):
    nomes = [n for n in EXPERIMENTOS if (RESULTS / "runs" / f"{n}_s{SEEDS[0]}.json").exists()]
    cols = 3
    lin = int(np.ceil(len(nomes) / cols))
    fig, axs = plt.subplots(lin, cols, figsize=(13, 3 * lin), sharey=True)
    for ax, n in zip(axs.ravel(), nomes):
        h = hist_medio(n)
        ep = np.arange(1, len(h["val_acc"]) + 1)
        ax.plot(ep, h["train_acc"], color=COR, label="treino")
        ax.plot(ep, h["val_acc"], color=COR2, label="validação")
        ax.set(title=n, xlabel="época")
    for ax in axs.ravel()[len(nomes):]:
        ax.axis("off")
    axs.ravel()[0].legend()
    axs[:, 0].tolist()[0].set_ylabel("acurácia")
    fig.tight_layout()
    if salvar:
        fig.savefig(RESULTS / "curvas" / "todas_acc.png")
    return fig


def val_vs_params(ag, salvar=True):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(ag.params / 1e6, ag.val_f1_mean, yerr=ag.val_f1_std, fmt="o", color=COR, capsize=3)
    for _, r in ag.iterrows():
        ax.annotate(r.exp, (r.params / 1e6, r.val_f1_mean), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set(xlabel="parâmetros (milhões)", ylabel="F1 macro no validation (média ± desvio)")
    fig.tight_layout()
    if salvar:
        fig.savefig(RESULTS / "curvas" / "val_vs_params.png")
    return fig


def matriz_confusao(salvar=True):
    cm = np.load(RESULTS / "test" / "confusao.npy")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="classe prevista", ylabel="classe real")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if salvar:
        fig.savefig(RESULTS / "test" / "matriz_confusao.png")
    return fig


def pares_confundidos(k=8):
    cm = np.load(RESULTS / "test" / "confusao.npy").copy()
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:k], cm.shape))[0]
    return [(int(a), int(b), int(cm[a, b])) for a, b in idx]


def grid_exemplos(n=6, salvar=True):
    y, p, conf = np.load(RESULTS / "test" / "pred_seed_principal.npy")
    y, p = y.astype(int), p.astype(int)
    df = le_split("test")
    rng = np.random.RandomState(0)
    acertos = rng.choice(np.where(y == p)[0], n, replace=False)
    erros = rng.choice(np.where(y != p)[0], n, replace=False)
    fig, axs = plt.subplots(2, n, figsize=(2.2 * n, 5))
    from PIL import Image
    from .utils import RAIZ
    for linha, idxs, tit in [(0, acertos, "acerto"), (1, erros, "erro")]:
        for ax, i in zip(axs[linha], idxs):
            im = Image.open(RAIZ / df.path[i]).convert("RGB").resize((LADO_RESIZE, LADO_RESIZE))
            o = (LADO_RESIZE - LADO) // 2
            ax.imshow(np.asarray(im)[o:o + LADO, o:o + LADO])
            ax.set_title(f"real {y[i]} / prev {p[i]}", fontsize=8,
                         color="green" if tit == "acerto" else "red")
            ax.axis("off")
    fig.tight_layout()
    if salvar:
        fig.savefig(RESULTS / "test" / "exemplos.png")
    return fig
