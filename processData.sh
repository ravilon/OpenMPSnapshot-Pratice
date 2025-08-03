#!/bin/bash
#2.1 Count directives:
 awk -f AWK/CountingDirectives.awk Jul2026/RepC/*/*.* > Jul2026PartialData/CountRepC.csv
 awk -f AWK/CountingDirectives.awk Jul2026/RepCPP/*/*.* > Jul2026PartialData/CountRepCPP.csv
#2.2 Count atomic clauses:
 awk -f AWK/CountingClausesAtomic.awk Jul2026/RepC/*/*.* Jul2026/RepCPP/*/*.*
#2.3 Count "Critical" with label:
 grep -E "^#pragma omp critical[ \t]*(\(.+\))" Jul2026/RepC*/*/*.* | wc -l
#2.4 Count atomic critical candidates:
 awk -f AWK/casosCritical.awk Jul2026/RepC*/*/*.*
#2.5 Count "schedule for":
 awk -f AWK/contaScheduleFor.awk Jul2026/RepC*/*/*.*
#2.6 Count "for" without "schedule":
 grep -E "^#pragma omp parallel for" Jul2026/RepC*/*/*.* | grep -v schedule  | wc -l
 grep -E "^#pragma omp  for" Jul2026/RepC*/*/*.* | grep -v schedule  | wc -l
#2.7 Count number of unbalanced loops:
 awk -f AWK/unbalancedLoop.awk Jul2026/RepC*/*/*.*
#2.8 Identify blocks with a sequence of "parallel for":
 awk -f AWK/closeParallelFor.awk Jul2026/RepC*/*/*.* | wc -l

