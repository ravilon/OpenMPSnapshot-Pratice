#include <stdio.h>
#include <string.h>
#include <omp.h>

#define MAX_LENGTH 1000 // Longitud máxima del string

int main() 
{
    char str[MAX_LENGTH];
    printf("Ingrese el string: ");
    fgets(str, sizeof(str), stdin);

    // Eliminar el salto de línea al final del string
    str[strcspn(str, "\n")] = '\0';

    int max_count = 0;
    char max_pair[3]; // El par de caracteres más el carácter nulo
    int n = strlen(str);
    
    // Inicializar variables para almacenar el máximo local de cada hilo
    int max_count_local = 0;
    char max_pair_local[3] = "";

    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) 
    {
        int count_local = 0;
        char pair[3] = {str[i], str[i + 1], '\0'}; // Formar el par de caracteres
        
        // Contar la cantidad de veces que aparece el par en la sección del string
        for (int j = 0; j < n - 1; j++) 
        {
            if (str[j] == pair[0] && str[j + 1] == pair[1]) 
            {
                count_local++;
            }
        }

        // Actualizar el máximo local si corresponde
        if (count_local > max_count_local) 
        {
            max_count_local = count_local;
            strcpy(max_pair_local, pair);
        }
    }

    // Encontrar el máximo global
    #pragma omp critical
    {
        if (max_count_local > max_count) 
        {
            max_count = max_count_local;
            strcpy(max_pair, max_pair_local);
        }
    }

    printf("El par que más se repite es \"%s\" con %d repeticiones.\n", max_pair, max_count);
    
    return 0;
}
