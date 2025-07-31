#include <iostream>
#include <algorithm>
#include "kd_tree.hpp"
#include <math.h>
#include <string>
#include <sstream>
#include <chrono>

#define N_DIM 2

//++++++++++++++++++        class kpoint        ++++++++++++++
template<typename T>
kpoint<T>::kpoint(){
    // points = {};
    ndim = N_DIM;
}

template<typename T>
T kpoint<T>::get_n_point(int axis){
    return points[axis];
}

template<typename T>
void kpoint<T>::set_point(int axis, T value){
    points[axis] = value;
}

template<typename T>
void kpoint<T>::print_kpoints(int axis){
    std::cout<<"( ";
    for(auto i=0;i<ndim;++i){
        std::cout<<points[i]<<", ";
    }
    std::cout<<"axis="<<axis<<") ";

    return;
}

template<typename T>
std::string kpoint<T>::save_kpoints(int axis){
    std::string s ("");
    s += "[";

    for(auto i=0;i<ndim;++i){
        s += std::to_string(points[i]);
        s += ",";
    }
    s += ";";
    s += std::to_string(axis);
    s += "]";

    return s;
}


//++++++++++++      class knode         +++++++++++++++++++++++++

template<typename T>
kdnode<T>::kdnode(){
    axis = -1; //indicates that the knode is empty
    left = NULL;
    right = NULL;
}

template<typename T>
void kdnode<T>::in_order(){

        if( axis != -1) {
            
            if ( left != NULL)
                left -> in_order();

            this -> split.print_kpoints(axis);

            if ( right != NULL)
                right -> in_order();

        }
    }

template<typename T>
void kdnode<T>::pre_order(){

    #ifdef DEBUG
        std::cout<<"Axis = "<<axis;
    #endif

        if( axis != -1) {
            
            this -> split.print_kpoints(axis);
            if (left != NULL)
                left -> pre_order();
            if (right != NULL)
                right -> pre_order();

        }
    }



//  ++++++++++       Other functions...      +++++++++++++++++++++

template<typename T>
void print_array(kpoint<T> &arr, int n, int axis){
    for(auto i=0; i<n; ++i){
        for(auto j=0; j<axis; ++j){
            std::cout<<arr.get_n_point(j)<<" ";
        }
        std::cout<<std::endl;
    }
}

template<typename T>
std::vector<struct kpoint<T>> choose_splitting_point(std::vector<struct kpoint<T>> points, int n, int myaxis){
    //use std::nth_element
    //your array won't be sorted, 
    //but you are only guaranteed instead that your nth element is in position n.

    // print_array(*points,n,myaxis);

    auto beg = points.begin();
    int half = n/2;

    std::nth_element(beg, beg + half, points.end(), 
        [&myaxis](kpoint<T>& a, kpoint<T>& b){return a.get_n_point(myaxis) < b.get_n_point(myaxis);}
    );


    return points; //return the first element of the sort array

}

template<typename T>
struct kdnode<T> * build_serial_kdtree( std::vector<struct kpoint<T>> points, int ndim, int axis ) {

/*
* points is a pointer to the relevant section of the data set;
* N is the number of points to be considered, from points to points+N
* ndim is the number of dimensions of the data points
* axis is the dimension used previsously as the splitting one
*/

    struct kdnode<T>* node = new kdnode<T>; 
    const int N = points.size();
    int myaxis = (axis+1) % ndim;

    if( N == 1 ) { //a leaf with the point *points;

        node->left = new kdnode<T>;
        node->right = new kdnode<T>;
        node->axis = myaxis;
        node->split = points.at(0);

    }else{

        auto mypoint = choose_splitting_point( points, N, myaxis);

        // the splitting point
        int half = N/2;

        //al posto di creare ogni volta, si potrebbe usare sempre lo stesso e prendere da una parte all'altra
        std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
        std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());
        //+1 so we can delet the splitting point

        node->axis = myaxis;

        node->split = mypoint.at(half);

        // to opt to save a pointer, instead
        node->left = build_serial_kdtree( left_points, ndim, myaxis );
        

        //if original size points is 2, is 1 for left, 1 for split and 0 for right;

        if( N != 2)
            node -> right = build_serial_kdtree( right_points, ndim, myaxis );
        else
            node -> right = new kdnode<T>;

    }

    return node;
    
}


template<typename T>
std::string serialize_node(struct kdnode<T> *kdtree){

    std::string s ("");

    try{

    if( kdtree -> axis != -1) {
            
            s = kdtree -> split.save_kpoints( kdtree -> axis);

            if ( kdtree -> left != NULL)
            s += "(" + serialize_node(kdtree -> left) + ")";
            
            if ( kdtree -> right != NULL)
            s += "(" + serialize_node(kdtree -> right) + ")";

        }
    }catch(const std::exception& e){
        std::cout<<e.what();
    }

    return s;
}

template<typename T>
struct kdnode<T> * deserialize_node(std::string data){

    std::string s = data;

    if ( s.size() == 0 )
        return new kdnode<T>; 
    
    if ( s[0] == ')' ) 
        return new kdnode<T>; 

    int j = 0;

    while ( j < s.size() && s[j] != '(' ) 
        j ++;

    T arr[N_DIM];
    std::string temp_str;
    if ( s[1] == '[' ){
        
        temp_str = s.substr(2, j-5);

    }else{
        // T arr1[N_DIM]{std::stoi(s.substr(1, j-5))}; //only the number of the array
        // for(int i =0; i<N_DIM; ++i)
        //     arr[i] = arr1[i];
        temp_str = s.substr(1, j-5);

    }

    int temp_val;
    std::istringstream iss(temp_str);
    for(int i =0; i<N_DIM; ++i) {
        iss >> temp_val;
        arr[i] = temp_val;
        if (iss.peek() == ',')
            iss.ignore();
    }
        
    struct kpoint<T> point(arr);
    struct kdnode<T> * root = new kdnode<T>;
    root -> split = point;

    temp_val = (int)s[j-2] - 48 ;//the numbers starts from 48 in the ASCII code
    root -> axis = temp_val; //where axis is saved

    int left = 0, i = j;

    // find separation between left and right definition
    while ( i < s.size() )  {

        if ( s[i] == '(' ) 
            left ++;
        else if ( s[i] == ')' ) 
            left --;

        if ( left == 0 ) {
            break;
        }
        i ++;
    }

    if ( j < s.size() - 1 ) {
        root->left = deserialize_node<T>(s.substr(j + 1, i - 1 - j));
    }
    if ( i + 1 < s.size() - 1 ) {
        root->right = deserialize_node<T>(s.substr(i + 2, s.size() - i - 2));   
    }
    return root;



}






// version 4.0
template<typename T>
struct kdnode<T> * build_parallel_kdtree4(std::vector<struct kpoint<T>> points, int ndim, int axis, int np, int level, MPI_Comm comm, int which){

    //IT ONLY ACCEPT NP == POWER OF 2 or POWER OF 2 + 1

    
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::chrono::duration<double> diff;


    int size, irank;
    // MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &irank);

    MPI_Status status;
    MPI_Request request;

    struct kdnode<T>* node = new kdnode<T>; 
    const int N = points.size();
    int myaxis = (axis+1) % ndim;

    if ( N == 1 ){

        #ifdef DEBUG
            // std::cout<<"\n Processor n: "<<irank<<", axis = "<<myaxis;
        #endif

        node->left = new kdnode<T>;
        node->right = new kdnode<T>;
        node->axis = myaxis;
        node->split = points.at(0);

    }else {
        

        auto mypoint = choose_splitting_point( points, N, myaxis);
        // the splitting point
        int half = N/2;
        node->axis = myaxis;
        node->split = mypoint.at(half);
        
        
        if ( irank != 0){

            //aspetta che arrivi il messaggio e da lì parte a fare l'albero
            //fa l'albero
            //manda a 0


            if ( np/2 != pow(2, level)) {

                //al posto di creare ogni volta, si potrebbe usare sempre lo stesso e prendere da una parte all'altra
                std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
                std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());
                //+1 so we can delet the splitting point

                // to opt to save a pointer, instead
                level = level + 1;
                //send var which same to left
                node->left = build_parallel_kdtree4( left_points, ndim, myaxis, np, level, comm, which);
                
                //sent var which + 1 to right
                which = which + 2;
                if( N != 2)
                    node -> right = build_parallel_kdtree4( right_points, ndim, myaxis, np, level, comm, which);
                else
                    node -> right = new kdnode<T>;



            }else{

                

                if( irank == which ){

                    //al posto di creare ogni volta, si potrebbe usare sempre lo stesso e prendere da una parte all'altra
                    std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
                    
                    #ifdef DEBUG
                        // std::cout<<"\tVector on 1: "<<left_points.size();
                    #endif

                        //return same level, so it will goes forever in the previous else
                    #pragma omp task shared(ndim) firstprivate(left_points, myaxis)
                    {
                        node -> left = build_serial_kdtree(left_points, ndim, myaxis);
                    }

                    // start = std::chrono::high_resolution_clock::now();
                    // #pragma omp task 
                    // {
                        std::string kdtree_str = serialize_node<T>(node -> left);
                        MPI_Send( kdtree_str.c_str() , kdtree_str.length() , MPI_CHAR , 0 , 10 , comm );
                    // }
                    // end = std::chrono::high_resolution_clock::now();
                    // diff = end - start;
                    // std::cout<<"ser:"<<diff.count()<<std::endl;
                    
                    
                    #ifdef DEBUG
                        std::cout<<"\n Processor n: "<<irank<<" AFTER SEND";
                    #endif
                }
                
                //sarebbe da distribuire al primo libero
                if ( which < np-1 ) // quindi si può fare
                    which = which + 1;
                else // se non si può fare, allora faccio lavorare un processore a caso, ovvero l'1
                    // l'1 perché dovrebbe essere il primo ad aver finito, quello con meno lavoro e così da non creare conflitti
                    which = 1;
                

                if( irank == which ){

                //al posto di creare ogni volta, si potrebbe usare sempre lo stesso e prendere da una parte all'altra
                std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());
                
                #ifdef DEBUG
                    // std::cout<<"\tVector on 2: "<<right_points.size();
                #endif
                //return same level, so it will goes forever in the previous else
                if( N != 2)
                    // node -> right = build_parallel_kdtree( right_points, ndim, myaxis, np, level, comm);
                    #pragma omp task shared(ndim) firstprivate(left_points, myaxis)
                    {
                        node -> right = build_serial_kdtree(right_points, ndim, myaxis);
                    }
                else
                    node -> right = new kdnode<T>;

                // #pragma omp task
                // {
                    std::string kdtree_str = serialize_node<T>(node -> right);
                    MPI_Send( kdtree_str.c_str() , kdtree_str.length() , MPI_CHAR , 0 , 20 , comm );  
                // }
                // start = std::chrono::high_resolution_clock::now();
                
                // end = std::chrono::high_resolution_clock::now();
                // diff = end - start;
                // std::cout<<"ser:"<<diff.count()<<std::endl;
                
                           
                
                #ifdef DEBUG
                    std::cout<<"\n Processor n: "<<irank<<" AFTER SEND";
                #endif
                }


                

            }



            // meanwhile ..

        } else if ( irank == 0) {
            
            if ( np/2 != pow(2, level)){

                //al posto di creare ogni volta, si potrebbe usare sempre lo stesso e prendere da una parte all'altra
                std::vector<struct kpoint<T>> left_points(mypoint.begin(), mypoint.begin() + half);
                std::vector<struct kpoint<T>> right_points(mypoint.begin() + half + 1, mypoint.end());
                //+1 so we can delet the splitting point

                // to opt to save a pointer, instead
                level = level + 1;
                node->left = build_parallel_kdtree4( left_points, ndim, myaxis, np, level, comm, which);
                
                which = which + 2; // possibly omit
                if( N != 2)
                    node -> right = build_parallel_kdtree4( right_points, ndim, myaxis, np, level, comm, which);
                else
                    node -> right = new kdnode<T>;




            }else{ //when it arrives at the good level

                #ifdef DEBUG
                    std::cout<<"\n Processor n "<<irank<<" BEFORE RECV ";
                #endif
                //recive from left (1)
                int flag = 0, count;

                //MPI_Probe for dynamic reciving size of the message
                //It's also possible to send a second message with the size, 
                // but it's faster now
                MPI_Probe(which, 10, comm, &status);  // Probe for an incoming message from process 1

                // When probe returns, the status object has the size and other
                // attributes of the incoming message. Get the message size
                MPI_Get_count( &status, MPI_CHAR, &count);
                // Allocate a buffer to hold the incoming numbers
                char *buf1 = new char[count];
                MPI_Recv(buf1, count, MPI_CHAR, which, 10, comm, &status);
                // MPI_Wait( &request , MPI_STATUS_IGNORE);
                std::string bla1(buf1, count);
                delete [] buf1;

                //calculate time
                // start = std::chrono::high_resolution_clock::now();
                node -> left = deserialize_node<T>(bla1);
                // end = std::chrono::high_resolution_clock::now();
                diff = end - start;
                // std::cout<<"des:"<<diff.count()<<std::endl;
                


                #ifdef DEBUG
                    std::cout<<"\n string "<<bla1<<" \nAFTER 1 RECV ";
                #endif
                
                
                //change which var
                if ( which < np-1 ) // quindi si può fare
                    which = which + 1;
                else // se non si può fare, allora faccio lavorare un processore a caso, ovvero l'1
                    // l'1 perché dovrebbe essere il primo ad aver finito, quello con meno lavoro e così da non creare conflitti
                    which = 1;
                
        
                //recive from right (2)
                flag = 0;
            
                MPI_Probe(which, 20, comm, &status);

                MPI_Get_count( &status, MPI_CHAR, &count); 
                char *buf2 = new char[count];
                MPI_Recv(buf2, count, MPI_CHAR, which, 20, comm, &status);  
                std::string bla2(buf2, count);
                delete [] buf2;

                // start = std::chrono::high_resolution_clock::now();
                node -> right = deserialize_node<T>(bla2);
                // end = std::chrono::high_resolution_clock::now();
                // diff = end - start;
                // std::cout<<"des:"<<diff.count()<<std::endl;

                //then recive the nodes done

                #ifdef DEBUG
                    std::cout<<"\n string "<<bla2<<" \nAFTER 2 RECV ";
                #endif

                


                

            }


        }



    }
    
    return node;

}



