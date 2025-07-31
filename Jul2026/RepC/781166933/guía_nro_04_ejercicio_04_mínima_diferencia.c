#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 100000

int main(int argc, char** argv)
{
    int A[N];
    int i, izquierda=0, derecha=0, diferencia=0;
    int minimo=0, tamanio=N, numero=0, aux, sumatotal=0;
    int *puntero;
    FILE *lee;
    FILE *esc;

    if(!(lee=fopen("entrada.ent","rb")))
    {
      printf("Error al abrir archivo.\n");
    }
    else
    {
      fscanf(lee,"%2d",&tamanio);

      //int A[tamanio];
      aux=fgetc(lee);
      
      puntero=&A[0];

      while(!feof(lee))
      {
        aux=fgetc(lee);
        if(aux!=32 && (aux>=48 && aux<=57))
        {
          aux-=48;
          numero+=aux;
          numero*=10;
        }
        else
        {
          *puntero = numero/10;
          sumatotal += *puntero;
          puntero++;
          numero=0;
        } 
      }
      minimo=sumatotal;
      puntero=&A[0];

      // Paralelización del bucle for con OpenMP
      #pragma omp parallel for // private(i, izquierda, derecha, diferencia) shared(minimo)
      for(i=0;i<tamanio-1;i++)
      {
        // Cada hilo calcula una parte de la suma izquierda
        izquierda += A[i];

        // Calculamos la suma derecha
        derecha = abs(sumatotal - izquierda);

        // Calculamos la diferencia
        diferencia = abs(derecha - izquierda);

        // Actualizamos el mínimo si es necesario
        #pragma omp critical
        {
            if(minimo > diferencia)
                minimo = diferencia;
        }
      }

      if(!(esc=fopen("salida.sal","wb")))
        printf("Error al guardar\n");
      else
        fprintf(esc,"%d ",minimo);
     
      fclose(lee);
      fclose(esc);
    }
}
