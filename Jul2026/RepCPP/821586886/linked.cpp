#include <iostream>
#include <omp.h>

#ifndef N
#define N 5
#endif

#ifndef FS
#define FS 25
#endif


struct node{
    int data;
    int fibdata;
    struct node* next;
};


int fib(int n){
    int x, y;
    if (n < 2){
        return (n);
    }
    else{

        #pragma omp task shared(x)
        {
            x = fib(n - 1);
        }

        #pragma omp task shared(y)
        {
            y = fib(n - 2);
        }

        #pragma omp taskwait
        return (x + y);
    }
}

void processwork(struct node* p){
   int n;
   n = p->data;
   p->fibdata = fib(n);
}


struct node* init_list(struct node* p){
    struct node* head = new node;
    struct node* temp = NULL;

    p = head;
    p->data = FS;
    p->fibdata = 0;

    for(int i=0; i<N; i++){
        temp = new node;
        p->next = temp;
        p = temp;
        p->data = FS+i+1;
        p->fibdata = i+1;
    }

    p->next = NULL;
    return head;
}


int main(){
    double start_time, run_time;

    struct node* p = NULL;
    struct node* temp = NULL;
    struct node* head = NULL;

    std::cout<<"Process linked list"<<std::endl;
    std::cout<<"Each linked list node will be processed by function 'processwork()'"<<std::endl;
    std::cout<<"Each ll node will compute "<<N<<" fibonacci numbers beginning with "<<FS<<std::endl;

    head = init_list(p);

    int n_threads=8;
    std::cout<<"Enter Number of Threads:";
    std::cin>>n_threads;
    omp_set_num_threads(8);

    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single nowait
        {   
            // Can also be done using while loop.
            for (p=head; p!=NULL; p = p->next){
                #pragma omp task firstprivate(p)
                {
                    processwork(p);
                }
            }
        }
    }

    run_time = omp_get_wtime() - start_time;

    p = head;
    while (p!=NULL){
        std::cout<< p->data <<":"<< p->fibdata <<std::endl;
        temp = p->next;
        delete p;
        p = temp;
    }

    std::cout<<"Compute Time: "<<run_time<<" seconds"<<std::endl;

    return 0;
}