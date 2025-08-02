#!/bin/bash

# This script finds and lists files in Jul2026/RepC and Jul2026/RepCPP directories
find Jul2026/RepC Jul2026/RepCPP \
  -type f ! \( -iname "*.c" -o -iname "*.h" -o -iname "*.C" -o -iname "*.H" -o -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cxx" -o -iname "*.hxx" -o -iname "*.inl" \) \
  -delete 

# Uncomment the following line to delete the files found by the above command
#find Jul2026/RepC Jul2026/RepCPP \
#  -type f ! \( -iname "*.c" -o -iname "*.h" -o -iname "*.C" -o -iname "*.H" -o -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cxx" -o -iname "*.hxx" -o -iname "*.inl" \) \
#  -delete