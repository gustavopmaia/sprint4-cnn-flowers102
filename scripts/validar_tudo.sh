#!/bin/bash
# roda da pasta sprint4: bash scripts/validar_tudo.sh
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export MPLBACKEND=Agg
falhas=0

item() {  # descricao, comando...
  desc="$1"; shift
  saida=$("$@" 2>&1); cod=$?
  if [ $cod -eq 0 ]; then echo "PASS $desc"; else echo "FAIL $desc"; echo "$saida" | tail -5; falhas=$((falhas+1)); fi
}

item "divisao identica a sprint 3, caminhos disjuntos, so os 2 duplicados de conteudo do dataset original" $PY scripts/verifica_split.py
item "seeds fixas: baseline re-treinado da os mesmos numeros" $PY scripts/checa.py reproduz
item ">= 3 arquiteturas em experiments.csv com params e val" $PY scripts/checa.py experimentos
item "test avaliado uma vez (lock) e so no modelo escolhido pelo validation" $PY scripts/checa.py lock_test
item "metricas e figuras minimas do test (matriz de confusao, exemplos)" $PY scripts/checa.py metricas_minimas
item "pdf com no maximo 8 paginas e tabela igual ao experiments.csv" $PY scripts/checa.py pdf
tmp=$(mktemp -d)
item "notebook executa do inicio ao fim" env -u MPLBACKEND $PY -m jupyter nbconvert --to notebook --execute notebook_experimentos.ipynb --output-dir "$tmp" --ExecutePreprocessor.timeout=-1

echo
if [ $falhas -eq 0 ]; then echo "tudo certo"; else echo "$falhas item(ns) falharam"; exit 1; fi
