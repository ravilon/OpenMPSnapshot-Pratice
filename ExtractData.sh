#!/bin/bash
set -euo pipefail

REPC_DIR="Jul2026/RepC"
REP_CPP_DIR="Jul2026/RepCPP"
OUT_DIR="Jul2026PartialData"
AWK_DIR="AWK"

mkdir -p "$OUT_DIR"

echo "==> [2.1] Counting directives (RepC / RepCPP)..."
awk -f "$AWK_DIR/CountingDirectives.awk" $REPC_DIR/*/*.* > "$OUT_DIR/CountRepC.csv"
awk -f "$AWK_DIR/CountingDirectives.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/CountRepCPP.csv"

echo "==> [2.1] Merging directive counts..."
cat "$OUT_DIR/CountRepC.csv" "$OUT_DIR/CountRepCPP.csv" > "$OUT_DIR/CountALL.csv"
TOTAL_DIRECTIVES=$(wc -l < "$OUT_DIR/CountALL.csv")

echo "==> [2.2] Counting atomic clauses..."
awk -f "$AWK_DIR/CountingClausesAtomic.awk" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* > "$OUT_DIR/ClausesAtomic.txt"

echo "==> [2.3] Counting 'critical' with label..."
CRITICAL_LABEL_COUNT=$(grep -E "^#pragma omp critical[ \t]*(\(.+\))" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* | wc -l)

echo "==> [2.4] Counting atomic critical candidates..."
awk -f "$AWK_DIR/casosCritical.awk" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* > "$OUT_DIR/CasosCritical.txt"
TOTAL_CASOS_CRITICAL=$(wc -l < "$OUT_DIR/CasosCritical.txt")

echo "==> [2.5] Counting 'schedule for'..."
awk -f "$AWK_DIR/contaScheduleFor.awk" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* > "$OUT_DIR/ScheduleForCount.txt"
TOTAL_SCHEDULE_FOR=$(wc -l < "$OUT_DIR/ScheduleForCount.txt")

echo "==> [2.6] Counting 'for' without 'schedule'..."
FOR_NO_SCHEDULE1=$(grep -E "^#pragma omp parallel for" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* | grep -v schedule | wc -l)
FOR_NO_SCHEDULE2=$(grep -E "^#pragma omp  for" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* | grep -v schedule | wc -l)

echo "==> [2.7] Counting unbalanced loops..."
awk -f "$AWK_DIR/unbalancedLoop.awk" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* > "$OUT_DIR/UnbalancedLoops.txt"
TOTAL_UNBALANCED=$(wc -l < "$OUT_DIR/UnbalancedLoops.txt")

echo "==> [2.8] Counting parallel-for blocks..."
PARALLEL_FOR_BLOCKS=$(awk -f "$AWK_DIR/closeParallelFor.awk" $REPC_DIR/*/*.* $REP_CPP_DIR/*/*.* | tee "$OUT_DIR/ParallelForBlocks.txt" | wc -l)

echo ""
echo "========================================================"
echo "✅ SUMÁRIO FINAL AGRUPADO"
echo "--------------------------------------------------------"
echo "→ Diretivas totais: $TOTAL_DIRECTIVES"
echo "→ Labeled criticals: $CRITICAL_LABEL_COUNT"
echo "→ Atomic critical candidates: $TOTAL_CASOS_CRITICAL"
echo "→ Clauses 'schedule for': $TOTAL_SCHEDULE_FOR"
echo "→ 'parallel for' sem schedule: $FOR_NO_SCHEDULE1"
echo "→ 'for' sem schedule: $FOR_NO_SCHEDULE2"
echo "→ Laços desbalanceados: $TOTAL_UNBALANCED"
echo "→ Blocos com sequência de parallel for: $PARALLEL_FOR_BLOCKS"
echo "========================================================"

# Exportar para CSV
SUMMARY_CSV="$OUT_DIR/Summary.csv"
echo "Categoria,Total" > "$SUMMARY_CSV"
echo "Diretivas totais,$TOTAL_DIRECTIVES" >> "$SUMMARY_CSV"
echo "Labeled criticals,$CRITICAL_LABEL_COUNT" >> "$SUMMARY_CSV"
echo "Atomic critical candidates,$TOTAL_CASOS_CRITICAL" >> "$SUMMARY_CSV"
echo "Clauses schedule for,$TOTAL_SCHEDULE_FOR" >> "$SUMMARY_CSV"
echo "parallel for sem schedule,$FOR_NO_SCHEDULE1" >> "$SUMMARY_CSV"
echo "for sem schedule,$FOR_NO_SCHEDULE2" >> "$SUMMARY_CSV"
echo "Laços desbalanceados,$TOTAL_UNBALANCED" >> "$SUMMARY_CSV"
echo "Blocos parallel for em sequência,$PARALLEL_FOR_BLOCKS" >> "$SUMMARY_CSV"

echo "✅ Arquivo de sumário salvo em: $SUMMARY_CSV"
