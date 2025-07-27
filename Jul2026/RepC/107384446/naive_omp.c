#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>

int compare(char * , char *);
int filelength(char *);
double findCount(char * , char *,int length);
int main() {
  //Intializing variables
  int posTotal, negTotal;
  float posProb,negProb,posCons,negCons,TotalPos,TotalNeg;
  double pos, neg ,k;
  char temp[10000];
  clock_t t;

  //Assigning Values
  t=clock();
  posTotal=filelength("positive.txt");
  negTotal=filelength("negative.txt");
  posCons=(posTotal*1.0)/(posTotal+negTotal);
  negCons=(negTotal*1.0)/(posTotal+negTotal);
  //printf("%d %d\n",posTotal,negTotal );
  ///*
  //Taking Input
  printf("Enter Sentence to be processed :");
  gets(temp,sizeof(temp),stdin);
  printf("%s \n",temp);


  //Evaluating Values
  char *token=strtok(temp," ");
  while(token != NULL){

    //finding probablities
      k = findCount(token,"positive.txt",posTotal);
      pos = log((k + 1)/(posTotal*1.0));
      k=findCount(token,"negative.txt",negTotal);
      neg = log((k + 1)/(negTotal*1.0));
      printf("%lf %lf\n",pos,neg);
      TotalPos+=pos;
      TotalNeg+=neg;
      token = strtok(NULL, " ");
    }

    //Calculating Final probablities
    TotalPos+=posCons;
    TotalNeg+=negCons;
    printf("%lf %lf\n",TotalPos,TotalNeg );
    t=clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;

    //printiting results
  if (TotalPos> TotalNeg )
    printf("positive Sentence\n" );
  else
    printf("Negative Sentence\n" );
    //*/
    printf("%lf\n",time_taken );
  return 0;
}


double findCount(char *a , char * b , int c){
  FILE *fp ;
  char str[10];
  char temp[20] = "files/";
  char temp2[10];
  int i=0,length=0, k=0;
  double count=0;
  length=strlen(temp);
  for(i=0;i<strlen(b);i++){
    temp[length+i]=b[i];
  }
  temp[length+i]='\0';
  fp= fopen(temp,"r");
  k=0;
  //printf("%s\n",temp );
  #pragma omp parallel for schedule(static,4) shared(count)
  for(i=0;i<c;i++){
    fscanf(fp,"%s",str);
      if (strcmp(a,str) == 0)
        count=count+1;
    }

  //printf("World\n" );
  return count;

}


int filelength(char *a ){
  FILE *fp ;
  char str[10];
  char temp[20] = "files/";
  int i=0,length=0,count=0;
  length=strlen(temp);
  #pragma omp parallel for shared(temp,a)
  for (i=0;i<strlen(a);i++){
    temp[length+i]=a[i];
  }
  temp[length+strlen(a)]='\0';
  //printf("%s\n",temp );
  fp= fopen(temp,"r");
  #pragma omp parallel
  {
    while(fscanf(fp,"%s",str)!=EOF){
      //printf("%s\n",str );
      if(count > 10500)
        break;
        #pragma omp task shared(count)
        count=count+1;
    }

  }
  #pragma omp taskwait
  /*for(i=0;i<strlen(a);i++){
    temp[length+i]=a[i];
  }
  temp[length+i]='\0';
  fp= fopen(temp,"r");
  while(fscanf(fp,"%s",str)!=EOF){
    if(count>11000)
      break;
    count++;
  }
  //printf("World\n" );*/
  fclose(fp);
  return count;
}


int compare(char *a,char *b){
  int i;
  for (i=0;i<strlen(a)-2;i++){
    if(a[i]!=b[i])
      return 0;
  }
  return 1;
}
