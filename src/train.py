import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from .data import loaders, normaliza
from .models import CNN, EXPERIMENTOS, conta_params, monta_cfg
from .utils import (CLIP, LR, MAX_EPOCAS, PACIENCIA, RESULTS, get_device, le_json,
                    salva_json, set_seed, versoes)


def roda_epoca(model, loader, device, opt=None):
    treino = opt is not None
    model.train(treino)
    perda_total, certos, n = 0.0, 0, 0
    preds, alvos = [], []
    with torch.set_grad_enabled(treino):
        for x, y in loader:
            x, y = normaliza(x, device), y.to(device)
            saida = model(x)
            perda = nn.functional.cross_entropy(saida, y)
            if treino:
                opt.zero_grad()
                perda.backward()
                if CLIP:
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            perda_total += perda.item() * len(y)
            p = saida.argmax(1)
            certos += (p == y).sum().item()
            n += len(y)
            preds.append(p.cpu())
            alvos.append(y.cpu())
    return perda_total / n, certos / n, torch.cat(preds).numpy(), torch.cat(alvos).numpy()


def metricas(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=float((y == p).mean()), precision=float(pr), recall=float(rc), f1=float(f1))


def diagnostico(h, melhor):
    tr, va = h["train_acc"][melhor], h["val_acc"][melhor]
    if tr < 0.5 and va < 0.5:
        return "underfitting"
    if tr - va > 0.25:
        return "overfitting"
    return "ok"


def treina(nome, seed, pasta=None, max_epocas=None, salva_modelo=True, verbose=True):
    pasta = pasta or RESULTS / "runs"
    saida = pasta / f"{nome}_s{seed}.json"
    if saida.exists():  # retoma de onde parou
        return le_json(saida)
    max_epocas = max_epocas or MAX_EPOCAS
    device = get_device()
    set_seed(seed)
    cfg = monta_cfg(nome)
    model = CNN(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train, val = loaders(seed)

    h = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    melhor, melhor_acc, sem_melhora, estado = 0, -1.0, 0, None
    t0 = time.time()
    for ep in range(max_epocas):
        tl, ta, _, _ = roda_epoca(model, train, device, opt)
        vl, va, _, _ = roda_epoca(model, val, device)
        for k, v in zip(h, [tl, ta, vl, va]):
            h[k].append(v)
        if verbose:
            print(f"{nome} s{seed} ep {ep + 1:02d} loss {tl:.3f} acc {ta:.3f} val_loss {vl:.3f} val_acc {va:.3f}", flush=True)
        if va > melhor_acc:  # early stopping pela acuracia do validation
            melhor_acc, melhor, sem_melhora = va, ep, 0
            estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                break
    tempo = time.time() - t0

    model.load_state_dict(estado)
    _, _, p, y = roda_epoca(model, val, device)
    res = dict(
        nome=nome, seed=seed, cfg=cfg, params=conta_params(model),
        melhor_epoca=melhor + 1, epocas_rodadas=len(h["val_loss"]),
        val=metricas(y, p), tempo_s=round(tempo, 1), hist=h,
        diag=diagnostico(h, melhor), versoes=versoes())
    if salva_modelo:
        ck = RESULTS / "modelos" / f"{nome}_s{seed}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(estado, ck)
    salva_json(res, saida)
    return res
