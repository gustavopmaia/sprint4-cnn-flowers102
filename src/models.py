import torch.nn as nn

# cada experimento muda uma coisa em relação ao baseline
BASE = dict(filtros=[16, 32, 64], kernel=3, pool="max", head="flatten",
            denso=128, bn=False)

EXPERIMENTOS = {
    "baseline": dict(
        muda="Arquitetura inicial",
        hip="Ponto de partida: 3 blocos Conv-ReLU-MaxPool, Flatten e Dense 128.",
        cfg={}),
    "prof_2": dict(
        muda="2 blocos conv (16-32)",
        hip="Com menos blocos o mapa fica maior antes do Flatten. Queremos ver se 3 blocos eram necessários.",
        cfg=dict(filtros=[16, 32])),
    "prof_4": dict(
        muda="4 blocos conv (16-32-64-128)",
        hip="Um bloco a mais dá campo receptivo maior e reduz o Flatten.",
        cfg=dict(filtros=[16, 32, 64, 128])),
    "largura": dict(
        muda="Filtros 32-64-128",
        hip="Mais filtros por bloco, mais padrões de cor e textura por camada.",
        cfg=dict(filtros=[32, 64, 128])),
    "kernel_5": dict(
        muda="Kernel 5x5",
        hip="Campo receptivo maior em cada camada, pode ajudar com pétalas grandes.",
        cfg=dict(kernel=5)),
    "stride": dict(
        muda="Conv com stride 2 no lugar do MaxPool",
        hip="Downsampling aprendido em vez de fixo, pra ver se o MaxPool joga informação fora.",
        cfg=dict(pool="stride")),
    "gap": dict(
        muda="GlobalAveragePooling no lugar do Flatten",
        hip="Corta quase todos os parâmetros da cabeça e pode reduzir overfitting.",
        cfg=dict(head="gap")),
    "denso_256": dict(
        muda="Dense 256",
        hip="Camada densa maior: mais capacidade ou só mais parâmetros?",
        cfg=dict(denso=256)),
    "gap_bn": dict(
        muda="GAP + BatchNorm (muda só o BatchNorm em relação ao gap)",
        hip="O BatchNorm foi mal com Flatten. Com a cabeça pequena do GAP talvez funcione. Comparar com o gap, não com o baseline.",
        cfg=dict(head="gap", bn=True)),
    "batchnorm": dict(
        muda="BatchNorm depois de cada conv",
        hip="Eixo separado. Deve estabilizar o treino e acelerar a convergência.",
        cfg=dict(bn=True)),
}


def monta_cfg(nome):
    cfg = dict(BASE)
    cfg.update(EXPERIMENTOS[nome]["cfg"])
    return cfg


class CNN(nn.Module):
    def __init__(self, filtros, kernel, pool, head, denso, bn, n_classes=102):
        super().__init__()
        camadas, entrada = [], 3
        for f in filtros:
            stride = 2 if pool == "stride" else 1
            camadas.append(nn.Conv2d(entrada, f, kernel, stride=stride, padding=kernel // 2))
            if bn:
                camadas.append(nn.BatchNorm2d(f))
            camadas.append(nn.ReLU())
            if pool == "max":
                camadas.append(nn.MaxPool2d(2))
            entrada = f
        self.features = nn.Sequential(*camadas)
        if head == "gap":
            self.pool_final = nn.AdaptiveAvgPool2d(1)
            entrada_denso = entrada
        else:
            self.pool_final = nn.Identity()
            lado = 224
            for _ in filtros:
                lado //= 2
            entrada_denso = entrada * lado * lado
        self.classificador = nn.Sequential(
            nn.Flatten(), nn.Linear(entrada_denso, denso), nn.ReLU(),
            nn.Linear(denso, n_classes))

    def forward(self, x):
        return self.classificador(self.pool_final(self.features(x)))


def conta_params(model):
    return sum(p.numel() for p in model.parameters())
