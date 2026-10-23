# Sprint 4: arquiteturas de CNN no Flowers102

Gabriel Fidalgo (RM563213), Pedro Lima (RM565461), Gustavo Maia (RM562240), Gustavo Rossi (RM566075)

Continuação da Sprint 3: classificação das 102 espécies do Flowers102, com a mesma divisão Train/Val/Test (oficial do torchvision, 1.020 / 1.020 / 6.149 imagens). Comparamos 10 configurações de CNN no Validation, escolhemos uma e avaliamos no Test uma única vez.

## Resultado

| Experimento | Mudança | Parâmetros | F1 macro Val (média ± desvio) | Acurácia Val |
|---|---|---|---|---|
| baseline | Arquitetura inicial | 6.459.398 | 0,347 ± 0,007 | 0,356 |
| prof_2 | 2 blocos conv (16-32) | 12.863.430 | 0,323 ± 0,009 | 0,338 |
| prof_4 | 4 blocos conv (16-32-64-128) | 3.321.990 | 0,341 ± 0,003 | 0,354 |
| largura | Filtros 32-64-128 | 12.951.590 | 0,341 ± 0,005 | 0,355 |
| kernel_5 | Kernel 5x5 | 6.501.126 | 0,296 ± 0,016 | 0,312 |
| stride | Conv com stride 2 no lugar do MaxPool | 6.459.398 | 0,306 ± 0,020 | 0,320 |
| gap | GlobalAveragePooling no lugar do Flatten | 45.062 | 0,206 ± 0,010 | 0,232 |
| denso_256 | Dense 256 | 12.895.110 | 0,347 ± 0,010 | 0,360 |
| gap_bn | GAP + BatchNorm (muda só o BatchNorm em relação ao gap) | 45.286 | 0,282 ± 0,016 | 0,316 |
| batchnorm | BatchNorm depois de cada conv | 6.459.622 | 0,015 ± 0,017 | 0,043 |

Escolhida: **prof_4**, pela regra "maior F1 médio no Validation e, entre os que ficam a menos de 1 desvio dele, menos parâmetros". No Test (6.149 imagens, 3 seeds do mesmo modelo): acurácia 0,300 ± 0,009, F1 macro 0,293 ± 0,004. Figuras e métricas em `results/test/`, apresentação em `slides/apresentacao.pdf`.

## Como reproduzir

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_all                # treina tudo (retoma se parar)
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebook_experimentos.ipynb
.venv/bin/python slides/build_slides.py
bash scripts/validar_tudo.sh                   # PASS/FAIL de cada item
```

O dataset é baixado pelo torchvision para `data/` (ou, se o download demorar, é só colocar o `102flowers.tgz` lá e rodar de novo). `src/data.py` gera `splits/*.csv` e `scripts/verifica_split.py` confere com a Sprint 3. O notebook pula as rodadas que já têm json em `results/runs/`. Num clone limpo, o `evaluate_test()` roda uma vez, grava `results/test_done.lock` e recusa uma segunda chamada. Para refazer o Test é preciso apagar o lock de propósito.

Versões: Python 3.14, torch 2.14.1, torchvision 0.29.1 (em `requirements.txt` e em cada json de `results/runs/`).

## Protocolo e ajustes de orçamento

- Treino em MPS (MacBook Air M5), sem precisar do Colab. A rodada de 1 época do baseline leva uns 4 s, e as 30 rodadas (10 configurações x 3 seeds) somaram 2,6 horas de treino.
- Igual para todos: Adam, lr 5e-4, batch 32, no máximo 50 épocas, parada antecipada pela acurácia do Validation (paciência 10), seeds 42, 43 e 44, mesma augmentation no Train.
- A primeira rodada completa usou lr 1e-3, parada por val_loss e 40 épocas. Nela o `batchnorm` e um `kernel_5` caíram no acaso, e a parada por val_loss cortava o treino com a acurácia ainda subindo. Fizemos um piloto pequeno (baseline, `kernel_5` e `batchnorm`, seed 42) com lr 1e-3, 1e-3 com clip de gradiente e 5e-4, ficamos com 5e-4 e refizemos tudo. Os resultados da primeira rodada ficam em `results/piloto_v1/` só pra registro.
- O `gap_bn` foi adicionado depois de ver o `batchnorm` falhar. Ele muda só o BatchNorm em relação ao `gap`.
- O BatchNorm com Flatten continuou sem treinar com lr 5e-4. Não investigamos a causa, então não dá para concluir nada sobre BN.
- A melhor época média ficou entre 38 e 47 de 50, então talvez desse para treinar mais.
- Sem `class_weight`: o Train tem 10 imagens por classe.

## Observações sobre os dados

- O notebook da Sprint 3 só redimensionava para 224, mas o relatório descreve resize 256 e crop 224. Seguimos o relatório.
- O Flowers102 original tem 2 pares de arquivos idênticos entre conjuntos (image_07307/image_07309, train e test, e image_08077/image_08067, val e test). Como a divisão tem que ser a da Sprint 3, não mexemos. O `verifica_split.py` só aceita esses 2 pares.
- Usamos o ID das classes, como na Sprint 3, sem nomes.

## Pastas

`src/` código, `splits/` caminhos de cada conjunto, `results/` resultados (`experiments.csv` com a média por configuração, `runs.csv` por rodada, `runs/*.json` com curvas, `curvas/`, `test/`), `slides/` gerador e PDF (feito com matplotlib, sem LibreOffice), `scripts/` verificações.
