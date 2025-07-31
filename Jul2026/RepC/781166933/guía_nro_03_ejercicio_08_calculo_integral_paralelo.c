#include <stdio.h>
#include <omp.h>

#define N 100000000 // Cantidad de intervalos (mientras mas grande valor N, mejor resultado de la integral por trapecio)

// Función a integrar
double funcion(double x) 
{
    return x * x; 
}

int main() 
{
    double a = 0.0; // Límite inferior de la integral
    double b = 1.0; // Límite superior de la integral
    int n = N; // Número de subdivisiones
    double h = (b - a) / n; // Ancho de cada subintervalo
    double integral = 0.0; // Valor inicial de la integral

    // Calcular la integral en paralelo usando el método del trapecio
    #pragma omp parallel for reduction(+:integral)
    for (int i = 1; i < n; i++) 
    {
        double x0 = a + i * h; // Extremo izquierdo del trapecio
        double x1 = a + (i + 1) * h; // Extremo derecho del trapecio
        double y0 = funcion(x0); // Valor de la función en x0
        double y1 = funcion(x1); // Valor de la función en x1

        integral += (y0 + y1) * h / 2.0; // Área del trapecio (integral).
    }

    printf("Valor de la integral: %lf\n", integral);

    return 0;
}
