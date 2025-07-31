#include <iostream>    
#include <string>      
#include <omp.h>       

using namespace std;   

struct No {
    string nomeArquivo; 
    No* proximo;        

    No(const string& nome) {
        nomeArquivo = nome;
        proximo = nullptr;
    }
};

int main() {
    const int N = 20; 

    No* inicio = new No("ARQUIVO-1.txt");               
    No* cauda = inicio;   

    for (int i = 2; i <= N; ++i) {                      
        cauda->proximo = new No("ARQUIVO" + to_string(i) + ".txt"); 
        cauda = cauda->proximo;                         
    }


    #pragma omp parallel                              // INICIA REGIÃO PARALELA (VÁRIAS THREADS PODEM EXECUTAR ISSO)
    {
        #pragma omp single nowait                     // APENAS UMA THREAD EXECUTA O BLOCO, SEM BARREIRA AO FINAL
        {
            No* p = inicio;          
            while (p) {                               // ENQUANTO HOUVER NÓS NA LISTA
                #pragma omp task firstprivate(p)      // CRIA UMA TASK QUE CAPTURA 'P' POR VALOR
                {
                    int tid = omp_get_thread_num();   // OBTÉM O ID DA THREAD QUE EXECUTA A TASK
                    #pragma omp critical
                    {
                        cout << "Thread " << tid << ": processou: " << p->nomeArquivo << endl;
                    }
                }
                p = p->proximo;                       
            }
            #pragma omp taskwait                      // AGUARDA TODAS AS TASKS GERADAS NESTA REGIÃO
        }
    }

    
    while (inicio) {                                  // ENQUANTO HOUVER NÓS PARA LIBERAR
        No* tmp = inicio;                             // GUARDA O NÓ ATUAL EM 'TMP'
        inicio = inicio->proximo;                     // MOVE 'INICIO' PARA O PRÓXIMO NÓ
        delete tmp;                                   // DESALOCA O NÓ GUARDADO EM 'TMP'
    }

    return 0;                                         
}