#!/bin/bash
set -euo pipefail

REPC_DIR="Jul2026/RepC"
REP_CPP_DIR="Jul2026/RepCPP"
OUT_DIR="Jul2026PartialData"
#REPC_DIR="May2023/RepC"
#REP_CPP_DIR="May2023/RepCPP"
#OUT_DIR="May2023PartialData"

AWK_DIR="AWK"

mkdir -p "$OUT_DIR"

echo "==> [2.1] Counting directives..."
awk -f "$AWK_DIR/CountingDirectives.awk" $REPC_DIR/*/*.* > "$OUT_DIR/CountRepC.csv"
awk -f "$AWK_DIR/CountingDirectives.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/CountRepCPP.csv"
cat "$OUT_DIR/CountRepC.csv" "$OUT_DIR/CountRepCPP.csv" > "$OUT_DIR/CountALL.csv"

TOTAL_DIRECTIVES_C=$(wc -l < "$OUT_DIR/CountRepC.csv")
TOTAL_DIRECTIVES_CPP=$(wc -l < "$OUT_DIR/CountRepCPP.csv")
TOTAL_DIRECTIVES=$((TOTAL_DIRECTIVES_C + TOTAL_DIRECTIVES_CPP))

echo "==> [2.2] Counting atomic clauses..."
awk -f "$AWK_DIR/CountingClausesAtomic.awk" $REPC_DIR/*/*.* > "$OUT_DIR/ClausesAtomicC.txt"
awk -f "$AWK_DIR/CountingClausesAtomic.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/ClausesAtomicCPP.txt"
TOTAL_CLAUSES_C=$(wc -l < "$OUT_DIR/ClausesAtomicC.txt")
TOTAL_CLAUSES_CPP=$(wc -l < "$OUT_DIR/ClausesAtomicCPP.txt")
TOTAL_CLAUSES=$((TOTAL_CLAUSES_C + TOTAL_CLAUSES_CPP))

echo "==> [2.3] Counting 'critical' with label..."
CRITICAL_LABEL_C=$(grep -E "^#pragma omp critical[ \t]*(\(.+\))" $REPC_DIR/*/*.* | wc -l)
CRITICAL_LABEL_CPP=$(grep -E "^#pragma omp critical[ \t]*(\(.+\))" $REP_CPP_DIR/*/*.* | wc -l)
CRITICAL_LABEL_TOTAL=$((CRITICAL_LABEL_C + CRITICAL_LABEL_CPP))

echo "==> [2.4] Counting atomic critical candidates..."
awk -f "$AWK_DIR/casosCritical.awk" $REPC_DIR/*/*.* > "$OUT_DIR/CasosCriticalC.txt"
awk -f "$AWK_DIR/casosCritical.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/CasosCriticalCPP.txt"
TOTAL_CASOS_C=$(wc -l < "$OUT_DIR/CasosCriticalC.txt")
TOTAL_CASOS_CPP=$(wc -l < "$OUT_DIR/CasosCriticalCPP.txt")
TOTAL_CASOS_CRITICAL=$((TOTAL_CASOS_C + TOTAL_CASOS_CPP))

echo "==> [2.5] Counting 'schedule for'..."
awk -f "$AWK_DIR/contaScheduleFor.awk" $REPC_DIR/*/*.* > "$OUT_DIR/ScheduleForC.txt"
awk -f "$AWK_DIR/contaScheduleFor.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/ScheduleForCPP.txt"
TOTAL_SCHED_C=$(wc -l < "$OUT_DIR/ScheduleForC.txt")
TOTAL_SCHED_CPP=$(wc -l < "$OUT_DIR/ScheduleForCPP.txt")
TOTAL_SCHED=$((TOTAL_SCHED_C + TOTAL_SCHED_CPP))

echo "==> [2.6] Counting 'for' without 'schedule'..."
FOR_NO_SCHED_C1=$(grep -E "^#pragma omp parallel for" $REPC_DIR/*/*.* | grep -v schedule | wc -l)
FOR_NO_SCHED_CPP1=$(grep -E "^#pragma omp parallel for" $REP_CPP_DIR/*/*.* | grep -v schedule | wc -l)
FOR_NO_SCHED_C2=$(grep -E "^#pragma omp  for" $REPC_DIR/*/*.* | grep -v schedule | wc -l)
FOR_NO_SCHED_CPP2=$(grep -E "^#pragma omp  for" $REP_CPP_DIR/*/*.* | grep -v schedule | wc -l)

FOR_NO_SCHED_C=$((FOR_NO_SCHED_C1 + FOR_NO_SCHED_C2))
FOR_NO_SCHED_CPP=$((FOR_NO_SCHED_CPP1 + FOR_NO_SCHED_CPP2))
FOR_NO_SCHED_TOTAL=$((FOR_NO_SCHED_C + FOR_NO_SCHED_CPP))

echo "==> [2.7] Counting unbalanced loops..."
awk -f "$AWK_DIR/unbalancedLoop.awk" $REPC_DIR/*/*.* > "$OUT_DIR/UnbalancedC.txt"
awk -f "$AWK_DIR/unbalancedLoop.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/UnbalancedCPP.txt"
UNBALANCED_C=$(wc -l < "$OUT_DIR/UnbalancedC.txt")
UNBALANCED_CPP=$(wc -l < "$OUT_DIR/UnbalancedCPP.txt")
UNBALANCED_TOTAL=$((UNBALANCED_C + UNBALANCED_CPP))

echo "==> [2.8] Counting parallel-for blocks..."
PARALLEL_FOR_C=$(awk -f "$AWK_DIR/closeParallelFor.awk" $REPC_DIR/*/*.* | tee "$OUT_DIR/ParallelForC.txt" | wc -l)
PARALLEL_FOR_CPP=$(awk -f "$AWK_DIR/closeParallelFor.awk" $REP_CPP_DIR/*/*.* | tee "$OUT_DIR/ParallelForCPP.txt" | wc -l)
PARALLEL_FOR_TOTAL=$((PARALLEL_FOR_C + PARALLEL_FOR_CPP))

# Criar CSV com sumário completo
SUMMARY_CSV="$OUT_DIR/Summary.csv"
echo "Categoria,RepC,RepCPP,Total" > "$SUMMARY_CSV"
echo "Diretivas totais,$TOTAL_DIRECTIVES_C,$TOTAL_DIRECTIVES_CPP,$TOTAL_DIRECTIVES" >> "$SUMMARY_CSV"
echo "Labeled criticals,$CRITICAL_LABEL_C,$CRITICAL_LABEL_CPP,$CRITICAL_LABEL_TOTAL" >> "$SUMMARY_CSV"
echo "Atomic critical candidates,$TOTAL_CASOS_C,$TOTAL_CASOS_CPP,$TOTAL_CASOS_CRITICAL" >> "$SUMMARY_CSV"
echo "Clauses schedule for,$TOTAL_SCHED_C,$TOTAL_SCHED_CPP,$TOTAL_SCHED" >> "$SUMMARY_CSV"
echo "parallel for sem schedule,$FOR_NO_SCHED_C,$FOR_NO_SCHED_CPP,$FOR_NO_SCHED_TOTAL" >> "$SUMMARY_CSV"
echo "for sem schedule,$FOR_NO_SCHED_C2,$FOR_NO_SCHED_CPP2,$((FOR_NO_SCHED_C2 + FOR_NO_SCHED_CPP2))" >> "$SUMMARY_CSV"
echo "Laços desbalanceados,$UNBALANCED_C,$UNBALANCED_CPP,$UNBALANCED_TOTAL" >> "$SUMMARY_CSV"
echo "Blocos parallel for em sequência,$PARALLEL_FOR_C,$PARALLEL_FOR_CPP,$PARALLEL_FOR_TOTAL" >> "$SUMMARY_CSV"

# Mostrar sumário
echo ""
echo "========================================================"
echo "✅ SUMÁRIO FINAL AGRUPADO"
echo "--------------------------------------------------------"
column -s, -t "$SUMMARY_CSV"
echo "========================================================"
echo "✅ Arquivo salvo: $SUMMARY_CSV"
