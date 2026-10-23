import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.plots import hist_medio  # noqa: E402
from src.utils import RESULTS, f3, fpar, le_json  # noqa: E402

AZUL, VERM, CINZA = "#1f4e79", "#c0504d", "#555555"
W, H = 13.333, 7.5
ag = pd.read_csv(RESULTS / "experiments.csv")
teste = le_json(RESULTS / "test" / "metricas.json")
sel = teste["modelo"]
linha = {r.exp: r for r in ag.itertuples()}


def slide(titulo):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.05, 0.92, titulo, fontsize=28, color=AZUL, weight="bold", va="center")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.865, 0.865], color=AZUL, lw=1.5))
    return fig


def texto(fig, x, y, s, size=15, cor="black", largura=60, weight="normal", espaco=1.5):
    linhas = []
    for par in s.split("\n"):
        linhas += textwrap.wrap(par, largura) or [""]
    fig.text(x, y, "\n".join(linhas), fontsize=size, color=cor, va="top", weight=weight, linespacing=espaco)
    return y - len(linhas) * size * espaco / 72 / H


def tabela(fig, x, y, larguras, linhas_, size=11, cabecalho=True, destaque=None, wraps=None):
    # larguras em fracao da figura; wraps = n de caracteres por linha de cada coluna
    y0 = y
    for i, lin in enumerate(linhas_):
        cel = [textwrap.wrap(str(c), wraps[j]) or [""] for j, c in enumerate(lin)]
        n = max(len(c) for c in cel)
        alt = (n * size * 1.25 + 8) / 72 / H
        if cabecalho and i == 0:
            fig.add_artist(plt.Rectangle((x - 0.005, y - alt), sum(larguras) + 0.005, alt, color=AZUL, transform=fig.transFigure))
        elif destaque and lin[0] == destaque:
            fig.add_artist(plt.Rectangle((x - 0.005, y - alt), sum(larguras) + 0.005, alt, color="#dce6f1", transform=fig.transFigure))
        cx = x
        for j, c in enumerate(cel):
            fig.text(cx, y - 4 / 72 / H, "\n".join(c), fontsize=size, va="top", linespacing=1.25,
                     color="white" if (cabecalho and i == 0) else "black",
                     weight="bold" if (cabecalho and i == 0) else "normal")
            cx += larguras[j]
        y -= alt
    fig.add_artist(plt.Line2D([x - 0.005, x + sum(larguras)], [y, y], color=CINZA, lw=0.8))
    return y


def imagem(fig, caminho, caixa):
    ax = fig.add_axes(caixa)
    ax.imshow(imread(caminho))
    ax.axis("off")
    return ax


def pm(m, s):
    return f"{f3(m)} ± {f3(s)}"


paginas = []

# 1
fig = slide("Classificando 102 espécies de flores")
texto(fig, 0.05, 0.82, "Modalidade: classificação de imagens. Cada imagem recebe uma única classe, entre as 102 espécies do Flowers102.", 17, largura=80)
texto(fig, 0.05, 0.68, "Entrada: imagem RGB de 224 x 224 pixels, já normalizada.\nSaída: um vetor com 102 valores, e a classe prevista é o maior deles.", 16, largura=80)
texto(fig, 0.05, 0.52, "Nesta sprint o foco é a arquitetura da CNN. Partimos de uma rede pequena e mudamos um fator por vez, sempre comparando no Validation. O Test só foi usado uma vez, no modelo escolhido.", 16, largura=85)
texto(fig, 0.05, 0.30, "Grupo", 15, AZUL, weight="bold")
texto(fig, 0.05, 0.26, "Gabriel Fidalgo - RM563213\nPedro Lima - RM565461\nGustavo Maia - RM562240\nGustavo Rossi - RM566075", 15, espaco=1.6)
paginas.append(fig)

# 2
fig = slide("Mesmo dataset e mesma divisão da Sprint 3")
y = tabela(fig, 0.05, 0.82, [0.12, 0.14, 0.30],
           [["Conjunto", "Imagens", "Observação"],
            ["Train", "1.020", "10 por classe"],
            ["Validation", "1.020", "10 por classe"],
            ["Test", "6.149", "de 20 a 238 por classe"]], size=14, wraps=[14, 12, 30])
texto(fig, 0.05, y - 0.04, "A divisão é a oficial do Flowers102 (a mesma que o torchvision entrega). Salvamos os caminhos em splits/*.csv e conferimos que bate com os números da Sprint 3 e que nenhum arquivo aparece em dois conjuntos. Pelo conteúdo, o dataset original tem 2 pares de imagens idênticas (train/test e val/test). Como a divisão não pode mudar, só registramos isso.", 14, largura=72)
texto(fig, 0.05, 0.385, "Pré-processamento", 16, AZUL, weight="bold")
texto(fig, 0.05, 0.345, "Resize para 256 x 256 e recorte de 224 x 224, depois normalização com média e desvio da ImageNet. Val e Test usam o recorte central. No Train o recorte é aleatório, com flip horizontal e vertical e rotação de até 30 graus. A Sprint 3 descrevia esse fluxo no relatório, mas o notebook dela só redimensionava direto para 224, então seguimos o relatório.", 14, largura=72)
texto(fig, 0.05, 0.14, "Não usamos class_weight: o Train é balanceado, então os pesos seriam todos iguais a 1.", 14, CINZA, largura=72)
import torch  # noqa: E402
from src.data import le_split  # noqa: E402
xt = torch.load(RAIZ / "data" / "cache_train_256.pt")
yt = le_split("train").label.values
ax = fig.add_axes([0.62, 0.35, 0.34, 0.45]); ax.axis("off")
grade = np.ones((2 * 224 + 8, 3 * 224 + 16, 3), dtype=np.uint8) * 255
for k, i in enumerate([0, 150, 300, 450, 600, 900]):
    r, c = divmod(k, 3)
    grade[r * 232:r * 232 + 224, c * 232:c * 232 + 224] = xt[i, :, 16:240, 16:240].permute(1, 2, 0).numpy()
ax.imshow(grade)
fig.text(0.79, 0.33, "exemplos do Train (6 classes diferentes)", ha="center", fontsize=10, color=CINZA)
paginas.append(fig)

# 3
fig = slide("Baseline: 3 blocos Conv-ReLU-MaxPool e Flatten")
blocos = [("Entrada\n224x224x3", "#eeeeee"), ("Conv 3x3\n16 filtros\nReLU\nMaxPool", "#cfe0f1"),
          ("Conv 3x3\n32 filtros\nReLU\nMaxPool", "#cfe0f1"), ("Conv 3x3\n64 filtros\nReLU\nMaxPool", "#cfe0f1"),
          ("Flatten\n28x28x64\n= 50.176", "#f1e3cf"), ("Dense 128\nReLU", "#f1e3cf"), ("Dense 102\nsaída", "#f1e3cf")]
ax = fig.add_axes([0.03, 0.50, 0.94, 0.30]); ax.axis("off"); ax.set_xlim(0, 7.4); ax.set_ylim(0, 1)
for i, (t, c) in enumerate(blocos):
    ax.add_patch(FancyBboxPatch((i * 1.06 + 0.04, 0.1), 0.92, 0.8, boxstyle="round,pad=0.01", fc=c, ec=CINZA))
    ax.text(i * 1.06 + 0.5, 0.5, t, ha="center", va="center", fontsize=11)
    if i < len(blocos) - 1:
        ax.annotate("", xy=(i * 1.06 + 1.08, 0.5), xytext=(i * 1.06 + 0.97, 0.5), arrowprops=dict(arrowstyle="->", color=CINZA))
b = linha["baseline"]
dense1 = 50176 * 128 + 128
texto(fig, 0.05, 0.42, f"{fpar(b.params)} parâmetros no total. Quase tudo está na primeira Dense: {fpar(dense1)} parâmetros ({dense1 / b.params * 100:.1f}%), porque o Flatten entrega 50.176 valores. Os três blocos de convolução juntos têm só {fpar(448 + 4640 + 18496)}.", 15, largura=95)
texto(fig, 0.05, 0.28, "Protocolo igual para todos os experimentos", 16, AZUL, weight="bold")
texto(fig, 0.05, 0.24, "Adam com lr 5e-4, batch 32, até 50 épocas, parada antecipada pela acurácia do Validation (paciência 10), 3 seeds (42, 43 e 44). Usamos lr 5e-4 porque com 1e-3 alguns modelos travaram no acaso logo na primeira época (testamos isso só em 3 configurações, seed 42).", 14, largura=100)
paginas.append(fig)

# 4
fig = slide("O que mudamos em cada experimento")
porque = {
    "baseline": ("Arquitetura inicial", "Ponto de partida"),
    "prof_2": ("2 blocos (16-32)", "Ver se o terceiro bloco era necessário"),
    "prof_4": ("4 blocos (16-32-64-128)", "Mais campo receptivo e Flatten menor"),
    "largura": ("Filtros 32-64-128", "Mais padrões por camada"),
    "kernel_5": ("Kernel 5x5", "Olhar uma região maior de cada vez"),
    "stride": ("Conv com stride 2, sem MaxPool", "Ver se o MaxPool joga informação fora"),
    "gap": ("GAP no lugar do Flatten", "Cortar parâmetros da cabeça"),
    "denso_256": ("Dense 256", "Mais capacidade ou só mais parâmetros?"),
    "batchnorm": ("BatchNorm depois de cada conv", "Estabilizar o treino"),
    "gap_bn": ("GAP + BatchNorm (muda só o BN em relação ao gap)", "O BN foi mal com Flatten, testamos com a cabeça pequena"),
}
rows = [["Experimento", "Mudança (uma por vez em relação ao baseline)", "Por que testamos"]]
for n in ag.exp:
    rows.append([n, porque[n][0], porque[n][1]])
y = tabela(fig, 0.05, 0.82, [0.14, 0.40, 0.36], rows, size=12, wraps=[16, 45, 42])
texto(fig, 0.05, y - 0.03, "O gap_bn foi adicionado depois de ver que o batchnorm não treinou. Ele é comparado com o gap, e não com o baseline.", 13, CINZA, largura=110)
paginas.append(fig)

# 5
fig = slide("Validation: quase tudo empata ou piora")
obs = {
    "baseline": "Ponto de partida, overfitting",
    "prof_2": "Piorou, e a cabeça dobrou de tamanho",
    "prof_4": "Empata com metade dos parâmetros",
    "largura": "Empata, mas treina bem mais devagar",
    "kernel_5": "Piorou",
    "stride": "Piorou um pouco",
    "gap": "Underfitting",
    "denso_256": "Empata, só mais parâmetros",
    "gap_bn": "Melhor que o gap, mas ainda abaixo",
    "batchnorm": "Não aprendeu com Flatten",
}
rows = [["Experimento", "Mudança principal", "Parâmetros", "F1 macro Validation (média ± desvio)", "Observação"]]
for r in ag.itertuples():
    rows.append([r.exp, porque[r.exp][0], fpar(r.params), pm(r.val_f1_mean, r.val_f1_std), obs[r.exp]])
curto = {"gap_bn": "GAP + BatchNorm (BN sobre o gap)", "batchnorm": "BatchNorm depois de cada conv"}
for lin in rows[1:]:
    lin[1] = curto.get(lin[0], lin[1])
y = tabela(fig, 0.03, 0.84, [0.10, 0.27, 0.10, 0.18, 0.31], rows, size=9.5, destaque=sel, wraps=[12, 38, 11, 22, 44])


def curva(box, nome, titulo):
    ax = fig.add_axes(box)
    h = hist_medio(nome)
    ep = np.arange(1, len(h["val_acc"]) + 1)
    ax.plot(ep, h["train_acc"], color=AZUL, label="treino")
    ax.plot(ep, h["val_acc"], color=VERM, label="validação")
    ax.set_title(titulo, fontsize=10); ax.set_xlabel("época", fontsize=9); ax.set_ylabel("acurácia", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1)


curva([0.06, 0.08, 0.24, 0.22], "baseline", "baseline: overfitting")
curva([0.38, 0.08, 0.24, 0.22], "gap", "gap: underfitting")
ax = fig.add_axes([0.70, 0.08, 0.26, 0.22])
ax.errorbar(ag.params / 1e6, ag.val_f1_mean, yerr=ag.val_f1_std, fmt="o", color=AZUL, capsize=2, ms=4)
ax.set_xlabel("parâmetros (milhões)", fontsize=9); ax.set_ylabel("F1 macro Val", fontsize=9)
ax.tick_params(labelsize=8); ax.grid(alpha=0.3)
paginas.append(fig)

# 6
fig = slide(f"Escolhemos o {sel}: 4 blocos, metade dos parâmetros")
s = linha[sel]
cand = ag[ag.val_f1_mean >= ag.val_f1_mean.max() - ag.loc[ag.val_f1_mean.idxmax(), "val_f1_std"]]
ax = fig.add_axes([0.07, 0.15, 0.42, 0.62])
d = ag.sort_values("val_f1_mean")
cores = [VERM if e == sel else AZUL for e in d.exp]
ax.barh(d.exp, d.val_f1_mean, xerr=d.val_f1_std, color=cores, capsize=2)
ax.set_xlabel("F1 macro no Validation (média ± desvio, 3 seeds)", fontsize=10); ax.tick_params(labelsize=10); ax.grid(axis="x", alpha=0.3)
y = texto(fig, 0.54, 0.80, f"Regra definida antes de rodar: pegamos o maior F1 macro médio e, entre os modelos a menos de 1 desvio dele, ficamos com o de menos parâmetros.", 14, largura=52)
nomes = ", ".join(cand.exp)
y = texto(fig, 0.54, y - 0.02, f"Ficaram empatados: {nomes}. O {sel} tem F1 {f3(s.val_f1_mean)} e {fpar(s.params)} parâmetros, contra {fpar(linha['baseline'].params)} do baseline (F1 {f3(linha['baseline'].val_f1_mean)}).", 14, largura=52)
y = texto(fig, 0.54, y - 0.02, "Por que faz sentido: mais um bloco de convolução diminui o mapa antes do Flatten (de 28x28 para 14x14), e a Dense 128 passa a ter metade dos pesos. O Flatten gigante era o que mais pesava no baseline. Ganhamos a mesma qualidade com um modelo menor.", 14, largura=52)
texto(fig, 0.54, y - 0.02, "Não dá para dizer que ele é melhor que o baseline, só que não é pior. Escolhemos pela simplicidade.", 14, CINZA, largura=52)
paginas.append(fig)

# 7
fig = slide("Avaliação final no Test")
ps = teste["por_seed"]
med = {k: np.mean([v[k] for v in ps.values()]) for k in ["acc", "precision", "recall", "f1"]}
dp = {k: np.std([v[k] for v in ps.values()]) for k in ["acc", "precision", "recall", "f1"]}
fig.texts[0].set_text(f"Test: acurácia de {f3(np.mean([v['acc'] for v in ps.values()]))} no {sel}")
rows = [["Métrica (6.149 imagens)", "seed 42", "média ± desvio das 3 seeds"]]
for k, nome in [("acc", "Acurácia"), ("precision", "Precision macro"), ("recall", "Recall macro"), ("f1", "F1 macro")]:
    rows.append([nome, f3(teste["principal"][k]), pm(med[k], dp[k])])
y = tabela(fig, 0.05, 0.82, [0.17, 0.08, 0.20], rows, size=12, wraps=[22, 10, 28])
import io  # noqa: E402
from src import plots  # noqa: E402
fg = plots.grid_exemplos(n=4, salvar=False)
for a in fg.axes:
    a.title.set_fontsize(12)
fg.set_size_inches(8.4, 4.6)
fg.subplots_adjust(hspace=0.25, top=0.93, bottom=0.02, left=0.02, right=0.98)
buf = io.BytesIO(); fg.savefig(buf, format="png", dpi=100); buf.seek(0); plt.close(fg)
imagem(fig, buf, [0.52, 0.42, 0.46, 0.40])
fig.text(0.75, 0.84, "Acertos (em cima) e erros (embaixo), seed 42", ha="center", fontsize=10, color=CINZA)
imagem(fig, RESULTS / "test" / "matriz_confusao.png", [0.03, 0.05, 0.30, 0.50])
from src.plots import pares_confundidos  # noqa: E402
pares = pares_confundidos(3)
pc = pd.read_csv(RESULTS / "test" / "por_classe.csv")
zeros = (pc.f1 == 0).sum()
texto(fig, 0.36, 0.34, f"No Validation o F1 era {f3(linha[sel].val_f1_mean)}, e no Test ficou em {f3(med['f1'])}. A queda é esperada, já que escolhemos épocas e modelo olhando o Validation.", 13, largura=50)
texto(fig, 0.36, 0.22, f"Pares mais confundidos (real/previsto): {pares[0][0]}/{pares[0][1]} ({pares[0][2]} imagens), {pares[1][0]}/{pares[1][1]} ({pares[1][2]}) e {pares[2][0]}/{pares[2][1]} ({pares[2][2]}). {zeros} classes tiveram F1 zero. Usamos o ID da classe, como na Sprint 3.", 13, largura=50)
texto(fig, 0.04, 0.55, "Matriz de confusão (102 x 102)", 10, CINZA, largura=40)
paginas.append(fig)

# 8
fig = slide("GAP foi o que mais pesou, e o overfitting é o problema")
g, b2 = linha["gap"], linha["baseline"]
texto(fig, 0.05, 0.82, "O que mais pesou", 17, AZUL, weight="bold")
texto(fig, 0.05, 0.78, f"Entre as mudanças que treinaram direito, trocar o Flatten pelo GAP teve o maior efeito: o F1 caiu de {f3(b2.val_f1_mean)} para {f3(g.val_f1_mean)} e o modelo ficou em underfitting, com só {fpar(g.params)} parâmetros. Perder a posição espacial das features, ou ficar pequeno demais, custou caro. O BatchNorm com Flatten caiu mais ainda (F1 {f3(linha['batchnorm'].val_f1_mean)}), mas foi falha de treino e não dá para concluir nada sobre BN. Kernel 5x5, stride no lugar do MaxPool e 2 blocos pioraram um pouco. Largura, Dense 256 e 4 blocos ficaram dentro da variação entre seeds.", 14, largura=100)
texto(fig, 0.05, 0.52, "Limitações", 17, AZUL, weight="bold")
texto(fig, 0.05, 0.48, f"Os modelos com Flatten fazem overfitting forte (acurácia de treino entre 0,672 e 0,871 contra 0,312 a 0,360 no Validation) e a melhor época caiu perto do limite de 50, então talvez ainda desse para treinar mais. Foram só 3 seeds por configuração, e com 10 imagens por classe a rede aprende pouco do zero. No Test, {zeros} classes ficaram com F1 zero, e isso não parece ter relação com o tamanho da classe.", 14, largura=100)
texto(fig, 0.05, 0.26, "Próximos passos", 17, AZUL, weight="bold")
texto(fig, 0.05, 0.22, "Testar transfer learning com uma rede pré-treinada na ImageNet, que é o caminho natural com tão poucas imagens por classe. Também queremos olhar dropout, weight decay e augmentation mais forte para atacar o overfitting, e repetir o BatchNorm com um lr menor.", 14, largura=100)
paginas.append(fig)

assert len(paginas) <= 8
saida = RAIZ / "slides" / "apresentacao.pdf"
with PdfPages(saida, metadata={"Title": "Sprint 4 - Arquiteturas de CNN no Flowers102", "Author": "Gabriel Fidalgo, Pedro Lima, Gustavo Maia, Gustavo Rossi", "Creator": "matplotlib", "Producer": "matplotlib"}) as pdf:
    for f in paginas:
        pdf.savefig(f)
plt.close("all")
print("slides:", len(paginas))
