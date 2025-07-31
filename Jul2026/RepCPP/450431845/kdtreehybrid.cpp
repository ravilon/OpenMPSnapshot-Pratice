#include<iostream>
#include<algorithm> // for sort
#include<vector> 
#include<bits/stdc++.h> // for 2D vector and sort function 
#include<random>
#include<mpi.h>
#include<omp.h>
using namespace std; 

const int NDIM= 2;

// a structure to represent node ok kd tree 

struct Node
{

	int point[NDIM]; // to store the dimensional point

	Node *left,*right; 
};



Node *kd_tree( std::vector<std::vector<double>> vect, bool myaxis, int* compt){
		
	
		struct Node *newnode = new Node;
	if (vect.size()==1){

		newnode-> point[0] = vect[0][0];
		newnode->point[1] =vect[0][1];
		newnode->right = newnode->left=NULL;
		*compt = *compt+1;
		return newnode;
	}
	else{ 
 			// let's find the median,
		int m=vect.size(); //number of row 
		int l= m/2;

	

		//1. sort vect according to axis  myaxis 
		
		if(myaxis==true){ // we sort according to y axis
//		#pragma omp parallel shared(vect,l,m)
//		{
			// 1. Let's swap the vector 
//				#pragma omp for ordered					
					for(int i=0; i<m; i++){
//						#pragma omp odered 
						swap(vect[i][0],vect[i][1]);}				
			// 2. sort the swap vector 

			sort(vect.begin(),vect.end());
			// 3. swap again the vector
//				#pragma omp for ordered 
				 
				for(int i=0; i<m; i++){
//					#pragma omp ordered 
					swap(vect[i][0],vect[i][1]);}
				
//		} // close pragma 
		}
		else{ // we sort according to x axis
			
			sort(vect.begin(),vect.end());
			}

		newnode->point[0]=vect[l][0];
		newnode->point[1] = vect[l][1];

		vector<vector<double>> left;
		vector<vector<double>> right; 

		//#pragma omp parellel shared (vect,l,m,left,right) 
		//{  
		
		for (int  i=0; i<l; i++)
			{
			left.push_back(vect[i]);}
		
		for(int i=l+1; i<m;i++)
			{
			right.push_back(vect[i]);}	

//		std:: cout<< " out "<< left.size()<<right.size() <<std::endl;
		#pragma omp parallel
		{	
			#pragma omp single nowait // Due to nowait claude, all the threads skip the 
//			//  implied barrier at the end of single region and wait here for being assigned a task
			{
			#pragma  omp task
				newnode->left = kd_tree(left,!myaxis, compt);
		#pragma omp task
		{
		if(right.size()>0) // this condition is use to avoid dumped core because, for 2 data, right=empty
			newnode->right= kd_tree(right,!myaxis, compt);
		}
			
		}
		}
		return newnode;
		
	}

}


int main(int argc, char **argv){

	int required; 

	omp_set_num_threads(16);
	MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE,&required);
	
	int rank =MPI::COMM_WORLD.Get_rank();
	int nprc= MPI::COMM_WORLD.Get_size();
	MPI_Status stats[nprc];

	int m=7;

	int n= pow(10,m);
	struct Node* root= new Node;
	vector<vector<double>> vect{};
	
	int compt{0};
	double tab[2];
	
	double starttime{0}, endtime{0};
	// we want to performe  10^m iterrations 	
	int count= n/nprc;
	int start = rank*count;
	int stop = start + count;
	bool myaxis=true;
	int th;	
	if(rank==0){
			#pragma omp parallel shared(vect,n, th)
			{
			int thrank= omp_get_thread_num(); // this return the id  of the thread
			int nthrds = omp_get_num_threads(); // return the number of thread  
			th = nthrds;
	// Now let's perform the for loop
			#pragma omp for  
			for(int i=0; i<n; i++){				 	
				srand(i);
				double a = double( rand())/double(RAND_MAX);
				srand(i+10);
				double b = double( rand())/double(RAND_MAX);
				#pragma omp critical	
				 	{ // we used critical to allow each thread to write on the shared 
				//memory, whithout its we will ends with dump core, because all the thread
				// will try to write on the same memory at the same times. 
				// so, this imply only one thread can be at the critical region, other
				// wait their turn. 
					vect.push_back({a,b});
					}
			}
	}
		starttime = MPI_Wtime();
		struct Node* ndo = new Node;
			sort(vect.begin(),vect.end()); // sort according to x axis.  
			int l= vect.size()/2;
		std::vector<std::vector<double>> vet{};
			
			MPI_Send(&l,1,MPI_INT,1,1,MPI_COMM_WORLD);
	
			for(int i=l+1;i<n;i++)
				{
			tab[0] = vect[i][0];
			tab[1] = vect[i][1];
		MPI_Send(&tab,2,MPI_DOUBLE,rank+1,rank+1,MPI_COMM_WORLD); // the second proc is the tag	
				}
			
		
			for(int i=0;i<l;i++)
				vet.push_back(vect[i]);
			
			root->point[0] = vect[l][0];
			root->point[1] = vect[l][1];

				vect ={};

			 l= vet.size()/2;
			MPI_Send(&l,1,MPI_INT,3,3,MPI_COMM_WORLD);
		
				for(int i=l+1;i<vet.size();i++)
					{
				tab[0] = vet[i][0];
				tab[1] = vet[i][1];
			MPI_Send(&tab,2,MPI_DOUBLE,rank+3,rank+3,MPI_COMM_WORLD); // the second proc is the tag	
					}
			
		
			for(int i=0;i<l;i++)
				vect.push_back(vet[i]);


			ndo->point[0] = vet[l][0];
			ndo->point[1] = vet[l][1];

			bool myaxis=true;		
			ndo->left = kd_tree(vect,myaxis, &compt); // normaly, it right should comme from process 3, since MPI is distributed memory,
	// it become difficul to link two process, we will have the same observation with other process.


			root->left = ndo;

			cout<< " Number of leaves  " << compt<<endl;
				endtime = MPI_Wtime();
			cout<< rank <<","<<th<<","<<m<<","<<endtime-starttime<<endl;
			
		}

	if(rank==1){
		starttime = MPI_Wtime();		
		int  l;
		
		MPI_Recv(&l,1,MPI_INT,rank-1,rank,MPI_COMM_WORLD,&stats[rank]);// receive from previous rank
		
		for(int i=1; i<l; i++){
			MPI_Recv(&tab,2,MPI_DOUBLE,rank-1,rank,MPI_COMM_WORLD,&stats[rank]);
			vect.push_back({tab[0],tab[1]});
			}
			
			 myaxis= true;
			
//			#pragma omp for ordered					
			for(int i=0; i<vect.size(); i++){
//					#pragma omp odered 
					swap(vect[i][0],vect[i][1]);}				
			// 2. sort the swap vector 

			sort(vect.begin(),vect.end());
			// 3. swap again the vector
	//			#pragma omp for ordered 
				 
			for(int i=0; i<vect.size(); i++){
//				#pragma omp ordered 
				swap(vect[i][0],vect[i][1]);}
	
		int h= vect.size()/2;
		struct Node* nd = new Node;
			nd->point[0] = vect[h][0];
			nd->point[1] = vect[h][0];
	
	
	MPI_Send(&h,1,MPI_INT,rank+1,rank+1,MPI_COMM_WORLD);
	
			for(int i=h+1;i<vect.size();i++)
				{
			tab[0] = vect[i][0];
			tab[1] = vect[i][1];
		MPI_Send(&tab,2,MPI_DOUBLE,rank+1,rank+1,MPI_COMM_WORLD); // the second proc is the tag	
				}

		std::vector<std::vector<double>> vet{};
		
		for(int i=0;i<h;i++)
			vet.push_back(vect[i]);
	
		nd->left = kd_tree(vet,myaxis, &compt);
		root->left=nd;
		cout<< " Number of leaves  " << compt<<endl;
			endtime = MPI_Wtime();
		cout<< rank<<","<<m<<","<<endtime-starttime<<endl;
		
		}// close if

	if(rank==3){
		starttime = MPI_Wtime();		
		int  l;
		
		MPI_Recv(&l,1,MPI_INT,0,rank,MPI_COMM_WORLD,&stats[rank]);// receive from previous rank
		
		for(int i=1; i<l; i++){
			MPI_Recv(&tab,2,MPI_DOUBLE,0,rank,MPI_COMM_WORLD,&stats[rank]);
			vect.push_back({tab[0],tab[1]});
			}
			
			 myaxis= true;
			
//			#pragma omp for ordered					
			for(int i=0; i<vect.size(); i++){
//					#pragma omp odered 
					swap(vect[i][0],vect[i][1]);}				
			// 2. sort the swap vector 

			sort(vect.begin(),vect.end());
			// 3. swap again the vector
	//			#pragma omp for ordered 
				 
			for(int i=0; i<vect.size(); i++){
//				#pragma omp ordered 
				swap(vect[i][0],vect[i][1]);}

		root = kd_tree(vect,myaxis, &compt);
		cout<< " Number of leaves  " << compt<<endl;
			endtime = MPI_Wtime();
		cout<< rank<<","<<m<<","<<endtime-starttime<<endl;
		
		}// close if


	if(rank==2 ){
		starttime = MPI_Wtime();		
		int  l;		
		MPI_Recv(&l,1,MPI_INT,1,2,MPI_COMM_WORLD,&stats[rank]);// receive from previous rank
		
		for(int i=1; i<l; i++){
			MPI_Recv(&tab,2,MPI_DOUBLE,1,rank,MPI_COMM_WORLD,&stats[rank]);
			vect.push_back({tab[0],tab[1]});
			}
			
		 myaxis = false;
		sort(vect.begin(),vect.end());

		root = kd_tree(vect,myaxis, &compt);
		cout<< " Number of leaves  " << compt<<endl;
			endtime = MPI_Wtime();
		cout<< rank<<","<<m<<","<<endtime-starttime<<endl;
		
		}
		
	MPI::Finalize();
	
	cout<< " Done .."<<endl;
		
	return 0;
	}
