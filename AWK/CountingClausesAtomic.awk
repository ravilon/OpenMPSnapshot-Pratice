#!/usr/bin/awk -f

BEGIN {
    semNada = 0
    update = 0
    write = 0
    read = 0
    hint = 0
    capture = 0
    compare = 0
    outros = 0
}

# Captura qualquer linha que começa com "#pragma omp atomic"
{
    if ($0 ~ /^#pragma[ \t]+omp[ \t]+atomic([ \t]|$)/) {
        # Remove comentários de linha e bloco (simples, sem tratar multi-linha)
        linha = $0
        sub("//.*", "", linha)
        sub("/\\*.*\\*/", "", linha)

        # Normaliza espaços
        gsub(/[ \t]+/, " ", linha)

        # Separa as palavras da linha
        split(linha, palavras, " ")
        tipo = ""
        for (i = 1; i <= length(palavras); i++) {
            if (palavras[i] == "atomic") {
                # tipo é a próxima palavra após "atomic", se houver
                tipo = palavras[i+1]
                break
            }
        }

        # Classificação dos tipos
        if (tipo == "" || tipo ~ /^$/) {
            semNada++
        } else if (tipo == "update") {
            update++
        } else if (tipo == "write") {
            write++
        } else if (tipo == "read") {
            read++
        } else if (tipo == "hint") {
            hint++
        } else if (tipo == "capture") {
            capture++
        } else if (tipo == "compare") {
            compare++
        } else {
            outros++
        }
    }
}

END {
    print "update: " update
    print "write: " write
    print "read: " read
    print "compare: " compare
    print "capture: " capture
    print "hint: " hint
    print "atomic puro: " semNada
    print "outros: " outros
    print "TOTAL: " semNada + update + write + read + compare + capture + hint + outros
}
