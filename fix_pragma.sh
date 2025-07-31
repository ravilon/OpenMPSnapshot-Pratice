#!/bin/bash
sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' Jul2026/RepC/*/*.*
sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n[[:space:]]*/ /; ba; }' Jul2026/RepCPP/*/*.*
grep -n '\\$' Jul2026/RepC/*/*.*
grep -n '\\$' Jul2026/RepCPP/*/*.*