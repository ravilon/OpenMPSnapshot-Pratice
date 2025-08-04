#!/bin/bash
set -euo pipefail

REP_CPP_DIR="May2023/RepCPP"
REPC_DIR="May2023/RepC"

echo "==> [1] Removendo espaços/tabs iniciais"
sed -i -E 's/^[ \t]+//' "$REPC_DIR"/*/*.*
sed -i -E 's/^[ \t]+//' "$REP_CPP_DIR"/*/*.*

echo "==> [2] Unindo diretivas #pragma quebradas em várias linhas"
sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' "$REPC_DIR"/*/*.*
sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' "$REP_CPP_DIR"/*/*.*

echo "==> [3] Verificando se restaram diretivas quebradas"
grep '\\$' "$REPC_DIR"/*/*.*
grep '\\$' "$REP_CPP_DIR"/*/*.*
