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

int main(){
    node nl;
    nl.left = NULL;
    nl.right = NULL;
    nl.data = 5;
    node nlr;
    nlr.left = NULL;
    nlr.right = NULL;
    nlr.data = 22;
    node nr;
    nr.left = &nlr;
    nr.right = NULL;
    nr.data = 10;
    node top;
    top.left = &nl;
    top.right = &nr;
    top.data = 44;

    int depth = 0;
    std::queue<node*> q;
    std::list<int> vis;

    q.push(&top);

    while(!q.empty()){
        int size = q.size();
        #pragma omp parallel for shared(q, vis, size) 
        for(int i=0; i<size; i++){
            node *c_n;
            #pragma omp critical
            {
                c_n = q.front();
                q.pop();
                vis.push_back(c_n->data);
            }
            #pragma omp critical            
            {
                if(c_n->left != NULL){
                    q.push(c_n->left);
                }      
                if(c_n->right != NULL) {
                    q.push(c_n->right);
                }
            }            
        }
    }
                

    for (auto const &i: vis) {
        printf("%d\n", i);
    }
}