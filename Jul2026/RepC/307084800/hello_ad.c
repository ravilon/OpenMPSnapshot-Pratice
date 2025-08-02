#include <omp.h>
#include <stdio.h>
int main() {
	//общая переменная, чтобы узнавать какой номер был напечатан предыдущим 
	int i = 0;
    #pragma omp parallel shared(i)
    {
    	//в цикле смотрим, пока каждый поток не напечатает свой номер
    	while (i != omp_get_num_threads()) {
    		// если у потока определенный номер, печатаем его, меняя i 
	    	if (i == omp_get_num_threads() - omp_get_thread_num() - 1) {
	    		printf("Hello, world! %d\n", omp_get_thread_num());
		    	#pragma omp atomic 
		    	i++;
	    	}
	    }
	}
}