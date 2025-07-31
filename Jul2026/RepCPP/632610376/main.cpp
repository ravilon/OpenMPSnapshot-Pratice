#include <iostream>

#include <omp.h>
#include <cmath>
#include <chrono>


const long long max = pow(10,8);


int main() {

    /***********************************************************
     * Varianta provedení sekvenčního výpočtu dvou smyček.
     * Žádné řízení přístupu ke sdílené proměnné není vyžadováno.
     ***********************************************************/
    auto start = std::chrono::steady_clock::now();
    long long sum = 0;
    for (long long i = 0 ; i < max;i++){
        sum++;
    }
    for (long long i = 0 ; i < max;i++){
        sum--;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Sequence operation elapsed time: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);



    /***********************************************************
     * Varianta provedení paralelního výpočtu dvou smyček,
     * kdy každá smyčka je jako celek vykonávána jedním vláknem.
     * Pro výpočet je využívána sdílená proměnná "sum" a není
     * použito žádné řízení přístupu k této proměnné. Díky tomu
     * je velké riziko, že dojde ke souběhu dvou vláken a tedy
     * k NESPRÁVNÉMU VÝSLEDKU operace.
     ***********************************************************/
    start = std::chrono::steady_clock::now();
    sum = 0;
    #pragma omp parallel shared(sum)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    sum++;
                }
            }
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    sum--;
                }
            }
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    std::cout << "No access control elapsed time: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);

    /***********************************************************
     * Varianta provedení paralelního výpočtu dvou smyček,
     * kdy každá smyčka je jako celek vykonávána jedním vláknem.
     * Pro výpočet je využívána sdílená proměnná "sum" a JE
     * použito žádné řízení přístupu v podobě atomických operací.
     * Díky tomu vznikne defakto sekvenční provádění operací, které
     * je navíc zpomaleno režijí v souvislosti s managementem vláken
     * a prováděním atomických operací.
     *
     * Výsledek bude SPRÁVNÝ ale pomalý.
     ***********************************************************/
    start = std::chrono::steady_clock::now();
    sum = 0;
    #pragma omp parallel shared(sum)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    #pragma omp atomic
                    sum++;
                }
            }
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    #pragma omp atomic
                    sum--;
                }
            }
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Access control by using atomic operations elapsed time: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);

    /***********************************************************
     * Varianta provedení paralelního výpočtu dvou smyček,
     * kdy každá smyčka je jako celek vykonávána jedním vláknem.
     * Pro výpočet je využívána sdílená proměnná "sum" a JE
     * použito žádné řízení přístupu v kritických sekcí.
     * Díky tomu vznikne defakto sekvenční provádění operací, které
     * je navíc zpomaleno režijí v souvislosti s managementem vláken
     * a prováděním zamykání mutexů.
     *
     * Výsledek bude SPRÁVNÝ ale VELMI pomalý.
     ***********************************************************/
    start = std::chrono::steady_clock::now();
    sum = 0;
    #pragma omp parallel shared(sum)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    #pragma omp critical
                    {
                        sum++;
                    }
                }
            }
            #pragma omp section
            {
                for (long long i = 0 ; i < max;i++){
                    #pragma omp critical
                    {
                        sum--;
                    }
                }
            }
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Access control by using critical seciton elapsed time: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);


    /***********************************************************
     * Lepší varianta paralelizace, kdy sice základem jsou dvě
     * paralelní sekce (každá smyčka předána k vykonávání jednomu
     * vláknu, ale díky vnořenému paralelizmu je dále každá
     * rozdělena mezi další dostupné vlákna. Dále díky použití
     * lokální proměnné a operace redukce nedochází k souběhu.
     *
     * Výsledek operace bude SPRÁVNÝ a nejspíše rychlejší než
     * předchozí varianty - přidělené vlákna jsou využity beze
     * zbytku.
     ***********************************************************/
    start = std::chrono::steady_clock::now();
    sum = 0;
    #pragma omp parallel shared(sum)
    {
    #pragma omp sections
        {
        #pragma omp section
            {
                long long tmpSum = 0;
                #pragma omp parallel for reduction(+:tmpSum)
                for (long long i = 0 ; i < max;i++){
                    tmpSum++;
                }
                #pragma omp atomic
                sum += tmpSum;
            }
            #pragma omp section
            {
                long long tmpSum = 0;
                #pragma omp parallel for reduction(+:tmpSum)
                for (long long i = 0 ; i < max;i++){
                        tmpSum--;
                    }
                #pragma omp atomic
                sum += tmpSum;
            }
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Nested loops elapsed time: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);


    /***********************************************************
     * Paralelizace, která nevyužívá samosattné sekce ale umisťuje
     * jednotlivé smyčky za sebe s využitím klíčového slova "nowait".
     * Díky tomu společně s využitím dvou oddělených sdílených
     * proměnných pro dané operace můžou probíhat jednotlivé otočky
     * obou cyklů souběžně tak, jak jsou volné jednotlivé výpočetní
     * vlákna.
     *
     ***********************************************************/
    start = std::chrono::steady_clock::now();
    sum = 0;
    long long tmpSumPlus = 0;
    long long tmpSumMinus = 0;
    #pragma omp parallel shared(sum, tmpSumPlus, tmpSumMinus)
    {
        #pragma omp for nowait reduction(+:tmpSumPlus)
                for (long long i = 0 ; i < max;i++){
                    tmpSumPlus++;
                }
        #pragma omp for nowait reduction(+:tmpSumMinus)
                for (long long i = 0 ; i < max;i++){
                    tmpSumMinus--;
                }
        #pragma omp barrier
        #pragma omp single
        {
            sum += tmpSumPlus;
            sum += tmpSumMinus;
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Sequential paralelized for loops: " << elapsed_seconds.count() << "s\n";
    printf("Result: %i\n", sum);



    return 0;
}
