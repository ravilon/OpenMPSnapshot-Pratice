#!/bin/bash

# Define os diretórios
REP_CPP_DIR="May2023/RepCPP"
REPC_DIR="May2023/RepC"

# Remove arquivos que não têm extensões de código-fonte nos diretórios definidos
find "$REPC_DIR" "$REP_CPP_DIR" \
  -type f ! \( -iname "*.c" -o -iname "*.h" -o -iname "*.C" -o -iname "*.H" -o -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cxx" -o -iname "*.hxx" -o -iname "*.inl" \) \
  -delete 

# Para usar com Jul2026, descomente e ajuste os diretórios:
#REP_CPP_DIR="Jul2026/RepCPP"
#REPC_DIR="Jul2026/RepC"
#find "$REPC_DIR" "$REP_CPP_DIR" \
#  -type f ! \( -iname "*.c" -o -iname "*.h" -o -iname "*.C" -o -iname "*.H" -o -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cxx" -o -iname "*.hxx" -o -iname "*.inl" \) \
#  -delete 
