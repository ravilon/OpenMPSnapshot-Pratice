#!/bin/bash
sed -i -E 's/^[ \t]+//' Jul2026/RepC/*/*.*
sed -i -E 's/^[ \t]+//' Jul2026/RepCPP/*/*.*
#sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' Jul2026/RepC/*/*.*
#sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' Jul2026/RepCPP/*/*.*
grep '\\$' Jul2026/RepC/*/*.*
grep '\\$' Jul2026/RepCPP/*/*.*