#!/bin/bash
set -euo pipefail

REP_C_DIR="Jul2026/RepC"
REP_CPP_DIR="Jul2026/RepCPP"
OUT_DIR="Jul2026PartialData"
AWK_DIR="AWK"

mkdir -p "$OUT_DIR"

echo "==> [2.1] Counting directives..."
awk -f "$AWK_DIR/CountingDirectives.awk" $REP_C_DIR/*/*.* > "$OUT_DIR/CountRepC.csv"
awk -f "$AWK_DIR/CountingDirectives.awk" $REP_CPP_DIR/*/*.* > "$OUT_DIR/CountRepCPP.csv"

echo "==> [2.2] Counting atomic clauses..."
awk -f "$AWK_DIR/CountingClausesAtomic.awk" $REP_C_DIR/*/*.* $REP_CPP_DIR/*/*.*

echo "==> [2.3] Counting 'critical' with label..."
grep -E "^#pragma omp critical[ \t]*(\(.+\))" $REP_C_DIR/*/*.* | wc -l

echo "==> [2.4] Counting atomic critical candidates..."
awk -f "$AWK_DIR/casosCritical.awk" $REP_C_DIR/*/*.* > "$OUT_DIR/CasosCritical.txt"

echo "==> [2.5] Counting 'schedule for'..."
awk -f "$AWK_DIR/contaScheduleFor.awk" $REP_C_DIR/*/*.* > "$OUT_DIR/ScheduleForCount.txt"

echo "==> [2.6] Counting 'for' without 'schedule'..."
FOR_NO_SCHEDULE1=$(grep -E "^#pragma omp parallel for" $REP_C_DIR/*/*.* | grep -v schedule | wc -l)
FOR_NO_SCHEDULE2=$(grep -E "^#pragma omp  for" $REP_C_DIR/*/*.* | grep -v schedule | wc -l)
echo "'parallel for' without schedule: $FOR_NO_SCHEDULE1"
echo "'for' without schedule: $FOR_NO_SCHEDULE2"

echo "==> [2.7] Counting unbalanced loops..."
awk -f "$AWK_DIR/unbalancedLoop.awk" $REP_C_DIR/*/*.* > "$OUT_DIR/UnbalancedLoops.txt"

echo "==> [2.8] Counting parallel-for blocks..."
awk -f "$AWK_DIR/closeParallelFor.awk" $REP_C_DIR/*/*.* | tee "$OUT_DIR/ParallelForBlocks.txt" | wc -l
