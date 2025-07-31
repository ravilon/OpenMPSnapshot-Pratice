#include "genetic.h"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <typeinfo>
#include <omp.h>
using namespace std;
/** @brief (one liner)
  *find the fitness of a chromosome
  * (documentation goes here)
  */
void Chromosome::evaluate()
{
//    Chromosome->fitness

//void evaluate(){
int n =(int)sqrt(this->representation.size());//square root of the full array which is n*n = n
int matrixRepresentation[n][n];
int offset=0,counter=0,a=0;//starting with offset 0 and counter 0

    for(int i = 0; i<n;i++){ //for each row in the Sudoku grid
        if(i>0){    //if we pass first row of the Sudoku
                if(i%(int)sqrt(n)==0){//if we are at the last row of a row of sub-grids
                        offset=n*i;//then the offset is n(number of elements in the row) * i(number of rows)
                        counter=offset;//save the offset in counter
                        }
                 else{                      //if we are not at the last row or a row of sub-grids
                            offset+=sqrt(n);//then we add sqrt(n) to the offset for the first sqrt n elements of the row
                                            //and later after printing each sqrt n elements the counter will increment by sqrt(n)
                            counter=offset; // save current value of the offset

                        }
                }
                a=0;
        for(int j = 0; j<sqrt(n); j++){//for each (sqrt(n)) sub-grids' row that is (sqrt(n))sized
            for(int k = 0; k<sqrt(n); k++){//for each element in that (sqrt(n))sized sub-grid's row
                matrixRepresentation[i][a]= this->representation.at(j+k+counter); //print the it based on it's sub-grid offset number
                                                         //added to the element number
                                                         //and the current assumed to be printed elements of the grid
 a++;
            }
                      counter+=((n-1));     //take into consideration the increment of the "printed square"
        }

    }
int rowFit=0;
int colFit=0;


for (int i = 0; i < n; i++)
{

int rowRep[n]={0};
int colRep[n]={0};

    for (int j = 0; j < n; j++)
    {
            if(rowRep[matrixRepresentation[i][j]-1]>=1)
                rowFit++;
            if(colRep[matrixRepresentation[j][i]-1]>=1)
                colFit++;
            rowRep[matrixRepresentation[i][j]-1]+=1;
            colRep[matrixRepresentation[j][i]-1]+=1;
    }

}

this->fitness = rowFit+colFit;

}

/** @brief (one liner)
  *
  * How to assign one chromosome to another using = operator
  * override of =
  */
Chromosome& Chromosome::operator=(const Chromosome& c)
{
    int n =sqrt(this->representation.size());//square root of the full array which is n*n = n

    for(int i = 0 ; i < n; i++) // for every element in this
            representation.at(i)= c.representation.at(i); //copy the values
    fitness = c.fitness;//copy the fitness

    return *this;
}

/** @brief (one liner)
  *
  * How to compare chromosomes to another using > operator
  * compare fitness
  */
bool Chromosome::operator > (const Chromosome b) {

         return(this->fitness > b.fitness);
      }

/** @brief (one liner)
  *
  * How to compare chromosomes to another using < operator
  * compare fitness
  */
bool Chromosome::operator < (const Chromosome b) {

         return (this->fitness < b.fitness);
      }

/** @brief (one liner)
  *
  * Constructor takes in the help array and seeds the empty slots with random numbers
  */
 Chromosome::Chromosome(int helpArray[],int size)
{
     srand((unsigned)time(0)*1000000); //just seeding the random function with the current system time

 this->representation=vector<int>(helpArray, helpArray+ size/ sizeof(int) );//using the helpArray as a guide

int n =sqrt(this->representation.size());//square root of the full array which is n*n = n

fitness=0;

 int randFiller;
 #pragma omp parallel for
for(int i =0 ; i < n; i++ ){                   //for each sub-grid

    for(int j=(i*n) ; j < ((i*n)+n); j++ ){     //for each element in the sub-grid

        if(representation.at(j)==0){            //needs to be filled
           randFiller=(rand()%n)+1;             //random number from 1->n

        for(int k=(i*n);k<(((i+1)*n));k++){
        if(representation.at(k)==randFiller){   //can't include this number as it already exists in it's sub-grid


            randFiller=(rand()%n)+1;            //find another random number
            k=(i*n)-1;                            //reset loop
        }
       }
       //#pragma omp critical
       representation.at(j)=randFiller;         //We can only reach here if the number does not exist in the sub-grid
}

        }
    }

evaluate();     //finding the fitness

}



/** @brief Deletes the pool of solutions (population)
  *
  * This is a function that makes sure to clear out memory
  * and avoids memory leaks
  */
void Population::clearPool(std::vector<Chromosome*>& pop)
{
  for(size_t i= 0 ;i < pop.size();i++)
    {
        delete pop[i];
    }

    pop.clear();
}

/** @brief Prints a chromosome (solution)
  *
  * Simple loop that loops the size of n*n Sudoku
  * based on the specified size
  */
void Population::printSolution(Chromosome chromo)
{
 int n =sqrt(chromo.representation.size());//square root of the full array which is n*n = n
/***
example 4*4 Sudoku:

        0  1  | 4  5
        2  3  | 6  7
       ------- ------
        8  9  | 12 13
        10 11 | 14 15
***/
 int offset=0,counter=0;//starting with offset 0 and counter 0
    for(int i = 0; i<n;i++){ //for each row in the Sudoku grid
        if(i>0){    //if we pass first row of the Sudoku
                if(i%(int)sqrt(n)==0){//if we are at the last row of a row of sub-grids
                        offset=n*i;//then the offset is n(number of elements in the row) * i(number of rows)
                        counter=offset;//save the offset in counter
                        for(int j=0;j<n;j++){//prints a line with the length of the row of the Sudoku
                            cout<<"___";
                            if((j+1)%(int)sqrt(n)==0 && j>0)
                                    cout <<" ";
                            }                   //ends printing the line that is at the last row of a row of sub-grids

                        cout<<endl;         //end line after the printed line (____\n)
                        }
                 else{                      //if we are not at the last row or a row of sub-grids
                            offset+=sqrt(n);//then we add sqrt(n) to the offset for the first sqrt n elements of the row
                                            //and later after printing each sqrt n elements the counter will increment by sqrt(n)
                            counter=offset; // save current value of the offset

                        }
                }
        for(int j = 0; j<sqrt(n); j++){//for each (sqrt(n)) sub-grids' row that is (sqrt(n))sized
            for(int k = 0; k<sqrt(n); k++){//for each element in that (sqrt(n))sized sub-grid's row
                cout <<" "<<chromo.representation.at(j+k+counter)<<" "; //print the it based on it's sub-grid offset number
                                                         //added to the element number
                                                         //and the current assumed to be printed elements of the grid
            }
                      cout <<"|";           //square vertical divider
                      counter+=((n-1));     //take into consideration the increment of the "printed square"
        }
        cout << endl;//end of sudoku row
    }
cout << "Fitness = "<< chromo.fitness;
}

/** @brief (one liner)
  *
  * (documentation goes here)
  */
Chromosome Population::getFittest()
{
  Chromosome a=genePool.at(0) ;
  int z=0;
  for(int i =1 ; i< populationSize;i++){
    if (genePool.at(i).fitness<a.fitness)
        z=i;
  }
  return genePool.at(z);


}



/** @brief (one liner)
  *
  *
  * Swap mutation
  */

void Population::mutation(int array[],vector <int> helpArray, int blockSize){
    srand(time(0)*100000);
int blockArray[blockSize];
int block;

int element1=rand()%blockSize;
int element2=rand()%blockSize;

while (true){
    block= rand()%(blockSize);

    for(int i =0; i< blockSize;i++){
        blockArray[i]=array[(block*blockSize)+i];
    }

    int counter = blockSize*blockSize;
while((helpArray.at((block*blockSize)+element1)!=0 || helpArray.at((block*blockSize)+element2)!=0 )){
    if(helpArray.at((block*blockSize)+element1)!=0)
    element1=rand()%blockSize;
    if(helpArray.at((block*blockSize)+element2)!=0)
    element2=rand()%blockSize;
counter--;
if (counter == 0) break;
}//what if its a full block and no zeros exist """FIX LOOP """
if(helpArray.at((block*blockSize)+element1)==0 && helpArray.at((block*blockSize)+element2)==0 ){
break;

}

}
int temp= (int)blockArray[element1];
blockArray[element1]=(int)blockArray[element2];
blockArray[element2]= temp;

for(int i = (block*blockSize);i<((block*blockSize)+blockSize);i++){

    array[i]=blockArray[i-(block*blockSize)];

}
}
/** @brief (one liner)
  *
  * finds the parents
  * creates next generation
  */
void Population::selectAndRecombine(int mutationRate,int elitisim){
    int n = sqrt(genePool.at(0).representation.size());
    vector<Chromosome> newGenePool,tempGenePool;
    srand(time(0));
    int totalFitness = 0;
    //first of all let's clear the roulette
    rouletteWheel.clear();

    Chromosome *Parent1,*Parent2;
    int Child1[n*n],Child2[n*n];
    for(int i = 0; i < (int)genePool.size(); i ++)
    {
        totalFitness += genePool.at(i).fitness;

    }

    double ratio = (double)totalFitness/(double)(RAND_MAX/2);

    int cFitness= 0;
    int prFitness = 0;
    for(int i =0 ; i< (int)genePool.size(); i++)
    {

            prFitness = cFitness;
            cFitness +=(genePool.at(i)).fitness;
          for(double j=(double)prFitness/ratio;j<(double)cFitness/ratio;j++)
            {

                rouletteWheel.push_back(i);


            }
    }
    int rouletteSize = rouletteWheel.size();
    for (int i=0; i<genePool.size(); i++)
        tempGenePool.push_back(genePool[i]);
        sort(tempGenePool.begin(),tempGenePool.end());
for(int t=0;t<elitisim;t++){
newGenePool.push_back(tempGenePool[t]);
}
    for( int i = elitisim; i < (int)genePool.size()-1; i +=2)//let's create a new population, the +=2 is since from 2 parents we get 2 children , so each run creates 2 new chromosomes
    {
int index =rouletteWheel.at( int((rouletteSize*rand())/(RAND_MAX + 1)) )  ;
        Parent1 = &genePool.at(index);


index =rouletteWheel.at( int((rouletteSize*rand())/(RAND_MAX + 1)) )  ;

            //random number from 0 to roulettesize-1
        Parent2 = &genePool.at(index);


srand(time(0)*10000);




for(int i =0;i<n;i++){
        //0 1 2 3 4 5 6 7 8
if(rand()%100 > 50){
        /// 0  1  2  3  4  5  6  7  8  | 9  10 11 12 13 14 15 16 17 | 18 19 20 21 22 23 24 25 26|
        /// 27 28 29 30 31 32 33 34 35 | 36 37 38 39 40 41 42 43 44 | 45 46 47 48 49 50 51 52 53|
        /// 54 55 56 57 58 59 60 61 62 | 63 64 65 66 67 68 69 70 71 | 72 73 74 75 76 77 78 79 80|

        /**
        0  1  2    9  10 11    18 19 20
        3  4  5    12 13 14    21 22 23
        6  7  8    15 16 17    24 25 26

        27 28 29   36 37 38    45 46 47
        30 31 32   39 40 41    48 49 50
        33 34 35   42 43 44    51 52 53

        54 55 56   63 64 65    72 73 74
        57 58 59   66 67 68    75 76 77
        60 61 62   69 70 71    78 79 80
        **/
    for(int j = i*n; j<(i*n)+n;j++){
        Child1[j]=Parent1->representation.at(j);
        Child2[j]=Parent2->representation.at(j);

    }
}else{
    for(int j = i*n; j<(i*n)+n;j++){
        Child1[j]=Parent2->representation.at(j);
        Child2[j]=Parent1->representation.at(j);

}

    }

}

srand(time(0)*10000);
int m=rand()%101;
 /**Mutation(Child1)**/
if(m<=mutationRate){

mutation(Child1,helpArray,n);

}
newGenePool.push_back(Chromosome(Child1,sizeof(Child1)));

srand(time(0)*10000);

m=rand()%101;
 /**Mutation(Child2)**/
if(m<=mutationRate){
mutation(Child1,helpArray,n);
}
newGenePool.push_back(Chromosome(Child2,sizeof(Child2)));
}

genePool.clear();
genePool=newGenePool;

    }



/** @brief (one liner)
  *
  * (documentation goes here)
  */
 Population::Population(int popSize, int hA[],int size){
     cout<< "pop const";
this->helpArray.resize(size);
    cout << "resized";
    srand(time(0)*100000); //just seeding the random function with the current system time
    for(int i =0; i < size; i++)
this->helpArray.at(i)=(hA[i]);
    cout << "filled up";
    this->populationSize = popSize;
cout<< "pop const";
    for(int k = 0; k < populationSize; k ++)
    {
        Chromosome a = Chromosome(hA,size);
        genePool.push_back(a); //populating the gene pool
    }

}
/** @brief (one liner)
  *
  * (documentation goes here)
  */


Genetic::Genetic()
{
    //ctor
}

Genetic::~Genetic()
{
    //dtor
}
