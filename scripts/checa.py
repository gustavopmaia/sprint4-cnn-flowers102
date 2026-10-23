import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.evaluate import LOCK, evaluate_test, seleciona  # noqa: E402
from src.train import treina  # noqa: E402
from src.utils import RESULTS, f3, fpar, le_json  # noqa: E402


def reproduz():
    # re-treina o baseline por 3 epocas num diretorio de teste e compara com o run salvo
    ref = le_json(RESULTS / "runs" / "baseline_s42.json")["hist"]
    with tempfile.TemporaryDirectory() as d:
        novo = treina("baseline", 42, pasta=Path(d), max_epocas=3, salva_modelo=False, verbose=False)["hist"]
    diffs = [abs(a - b) for a, b in zip(ref["val_loss"][:3], novo["val_loss"][:3])]
    return max(diffs) < 1e-4, f"diferenca maxima de val_loss nas 3 primeiras epocas: {max(diffs):.2e}"


def experimentos():
    ag = pd.read_csv(RESULTS / "experiments.csv")
    ok = len(ag) >= 3 and ag.params.notna().all() and ag.val_f1_mean.notna().all()
    return ok, f"{len(ag)} arquiteturas registradas"


def lock_test():
    import json
    if not LOCK.exists():
        return False, "test_done.lock nao existe"
    lock = json.loads(LOCK.read_text())
    met = le_json(RESULTS / "test" / "metricas.json")
    esperado = seleciona()
    if not (lock["modelo"] == met["modelo"] == esperado):
        return False, f"lock={lock['modelo']} metricas={met['modelo']} validation={esperado}"
    try:
        evaluate_test()
        return False, "segunda chamada de evaluate_test nao foi recusada"
    except RuntimeError:
        return True, f"modelo avaliado no test: {esperado}, segunda chamada recusada"


def pdf():
    pdf_path = RAIZ / "slides" / "apresentacao.pdf"
    info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    paginas = int([l for l in info.splitlines() if l.startswith("Pages:")][0].split()[1])
    if paginas > 8:
        return False, f"{paginas} paginas"
    texto = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True).stdout
    ag = pd.read_csv(RESULTS / "experiments.csv")
    faltando = []
    for _, r in ag.iterrows():
        for v in [f3(r.val_f1_mean), fpar(r.params)]:
            if v not in texto:
                faltando.append(f"{r.exp}:{v}")
    return not faltando, f"{paginas} paginas; valores faltando na tabela: {faltando or 'nenhum'}"


def metricas_minimas():
    met = le_json(RESULTS / "test" / "metricas.json")
    chaves = all(k in met["principal"] for k in ["acc", "precision", "recall", "f1"])
    arqs = ["matriz_confusao.png", "exemplos.png", "por_classe.csv", "confusao.npy"]
    faltam = [a for a in arqs if not (RESULTS / "test" / a).exists()]
    return chaves and not faltam, f"arquivos faltando: {faltam or 'nenhum'}"


if __name__ == "__main__":
    nome = sys.argv[1]
    ok, msg = globals()[nome]()
    print(("PASS" if ok else "FAIL"), nome, "-", msg)
    sys.exit(0 if ok else 1)
