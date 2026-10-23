import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support)

from .data import loader_test, normaliza
from .models import CNN, EXPERIMENTOS, monta_cfg
from .train import metricas
from .utils import RESULTS, SEEDS, get_device, le_json, salva_json

LOCK = RESULTS / "test_done.lock"


def agrega():
    # junta os json de cada rodada em runs.csv e experimentos.csv (media e desvio entre seeds)
    linhas = []
    for nome in EXPERIMENTOS:
        for s in SEEDS:
            f = RESULTS / "runs" / f"{nome}_s{s}.json"
            if f.exists():
                r = le_json(f)
                linhas.append(dict(
                    exp=nome, seed=s, params=r["params"], melhor_epoca=r["melhor_epoca"],
                    epocas=r["epocas_rodadas"], tempo_s=r["tempo_s"], diag=r["diag"],
                    **{f"val_{k}": v for k, v in r["val"].items()},
                    train_acc=r["hist"]["train_acc"][r["melhor_epoca"] - 1],
                    muda=EXPERIMENTOS[nome]["muda"], hip=EXPERIMENTOS[nome]["hip"]))
    runs = pd.DataFrame(linhas)
    runs.to_csv(RESULTS / "runs.csv", index=False)
    g = runs.groupby("exp", sort=False)
    ag = g.agg(muda=("muda", "first"), params=("params", "first"), n_seeds=("seed", "count"),
               val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", lambda x: x.std(ddof=0)),
               val_precision_mean=("val_precision", "mean"), val_recall_mean=("val_recall", "mean"),
               val_f1_mean=("val_f1", "mean"), val_f1_std=("val_f1", lambda x: x.std(ddof=0)),
               melhor_epoca_mean=("melhor_epoca", "mean"), tempo_s_mean=("tempo_s", "mean"),
               train_acc_mean=("train_acc", "mean"), hip=("hip", "first"),
               diag=("diag", lambda x: x.mode().iloc[0])).reset_index()
    ag.to_csv(RESULTS / "experiments.csv", index=False)
    return ag


def seleciona(ag=None):
    # regra definida antes de rodar: maior F1 macro medio no val; quem fica a menos de
    # 1 desvio do melhor empata, e entre os empatados fica o de menos parametros
    if ag is None:
        ag = pd.read_csv(RESULTS / "experiments.csv")
    top = ag.loc[ag.val_f1_mean.idxmax()]
    cand = ag[ag.val_f1_mean >= top.val_f1_mean - top.val_f1_std]
    return cand.sort_values(["params", "val_f1_mean"], ascending=[True, False]).iloc[0].exp


def evaluate_test():
    # so roda uma vez. Escolhe o modelo sozinha pelo validation.
    if LOCK.exists():
        raise RuntimeError("o test ja foi avaliado (test_done.lock existe)")
    nome = seleciona()
    device = get_device()
    loader, df = loader_test()
    cfg = monta_cfg(nome)
    por_seed = {}
    for s in SEEDS:
        model = CNN(**cfg).to(device)
        model.load_state_dict(torch.load(RESULTS / "modelos" / f"{nome}_s{s}.pt"))
        model.eval()
        ps, ys, ls = [], [], []
        with torch.no_grad():
            for x, y in loader:
                lg = model(normaliza(x, device)).cpu()
                ps.append(lg.argmax(1)); ys.append(y)
                ls.append(torch.softmax(lg, 1).max(1).values)
        por_seed[s] = (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(ls).numpy())
    LOCK.write_text(json.dumps(dict(modelo=nome, seeds=SEEDS, quando=datetime.now().isoformat()), indent=1))

    y, p, conf = por_seed[SEEDS[0]]
    out = RESULTS / "test"
    out.mkdir(parents=True, exist_ok=True)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(102), zero_division=0)
    pd.DataFrame(dict(classe=range(102), precision=pr, recall=rc, f1=f1, suporte=sup)).to_csv(out / "por_classe.csv", index=False)
    np.save(out / "confusao.npy", confusion_matrix(y, p, labels=range(102)))
    np.save(out / "pred_seed_principal.npy", np.stack([y, p, conf]))
    salva_json(dict(
        modelo=nome, seed_principal=SEEDS[0],
        principal=metricas(y, p),
        por_seed={str(s): metricas(v[0], v[1]) for s, v in por_seed.items()},
        n_test=len(y)), out / "metricas.json")
    return nome
