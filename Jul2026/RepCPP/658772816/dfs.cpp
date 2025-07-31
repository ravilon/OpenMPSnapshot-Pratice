#include <omp.h>
#include <stdio.h>
#include <queue>
#include <list>

class node
{
 public:
    node *left, *right;
    int data;
};

void traverse(std::list<int> *vis, node *nd){
    if(nd->left != NULL){
        #pragma omp task
        traverse(vis, nd->left);
    }
    
    #pragma omp taskwait
    #pragma omp critical
    {
        vis->push_back(nd->data);
    }
    
    if(nd->right != NULL){
        #pragma omp task
        traverse(vis, nd->right);
    }
}

int main(){
    node nl2;
    nl2.left = NULL;
    nl2.right = NULL;
    nl2.data = 78;
    node nl;
    nl.left = NULL;
    nl.right = &nl2;
    nl.data = 5;
    node nlr;
    nlr.left = NULL;
    nlr.right = NULL;
    nlr.data = 22;
    node nrr;
    nrr.left = NULL;
    nrr.right = NULL;
    nrr.data = 11;
    node nr;
    nr.left = &nlr;
    nr.right = &nrr;
    nr.data = 10;
    node top;
    top.left = &nl;
    top.right = &nr;
    top.data = 44;

    std::list<int> vis;

    #pragma omp parallel firstprivate(nl) shared(vis) num_threads(8)
    {
        #pragma omp single
        traverse(&vis, &top);
    }

    for (auto const &i: vis) {
        printf("%d\n", i);
    }
}