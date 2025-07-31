#include "Block.h"
#include "sha256.h"
#include <omp.h>   // Incluir o cabeçalho OpenMP
#include <iostream> // Incluir o cabeçalho para cout
#include <cstring>  // Incluir o cabeçalho para funções de string como strncpy

Block::Block(uint32_t nIndexIn, const std::string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn) {
    _nNonce = 0;
    _tTime = time(nullptr);
    sHash = _CalculateHash();
}

void Block::MineBlock(uint32_t nDifficulty) {
    char cstr[nDifficulty + 1];
    for (uint32_t i = 0; i < nDifficulty; ++i) {
        cstr[i] = '0';
    }
    cstr[nDifficulty] = '\0';

    bool found = false;
    char local_sPrevHash[33]; // Tamanho fixo para sPrevHash
    strncpy(local_sPrevHash, sPrevHash.c_str(), 32);
    local_sPrevHash[32] = '\0';

    char local_sData[_sData.size() + 1]; // Tamanho fixo para _sData
    strncpy(local_sData, _sData.c_str(), _sData.size());
    local_sData[_sData.size()] = '\0';

    char local_sHash[65]; // Tamanho fixo para local_sHash
    local_sHash[0] = '\0'; // Inicializa como string vazia

    // Variáveis locais para serem mapeadas explicitamente
    uint32_t local_nIndex = _nIndex;
    time_t local_tTime = _tTime;

    char str[nDifficulty + 1]; // Converter str para char[]
    strncpy(str, cstr, nDifficulty);
    str[nDifficulty] = '\0';

    // Ajuste o tamanho de hash para adequar ao map
    #pragma omp target map(tofrom: found, local_sHash) map(to: local_nIndex, local_sPrevHash, local_tTime, local_sData, str) // Diretiva OpenMP para GPU
    {
        std::string hash;
        uint32_t local_nonce;
        while (!found) {
            #pragma omp atomic capture
            local_nonce = ++_nNonce;

            hash = sha256(std::to_string(local_nIndex) + local_sPrevHash + std::to_string(local_tTime) + local_sData + std::to_string(local_nonce));

            if (strncmp(hash.c_str(), str, nDifficulty) == 0) {
                #pragma omp critical
                {
                    if (!found) {
                        found = true;
                        strncpy(local_sHash, hash.c_str(), 64); // Copia hash para local_sHash
                    }
                }
            }
        }
    } // Fim da região de destino para GPU

    // Atualizar sHash fora da região target usando local_sHash
    local_sHash[64] = '\0'; // Garante que local_sHash seja terminado corretamente
    sHash = local_sHash;

    // Saída para console fora da região target
    if (found) {
        std::cout << "Block mined: " << sHash << std::endl;
    }
}

inline std::string Block::_CalculateHash() const {
    std::stringstream ss;
    ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;
    return sha256(ss.str());
}
