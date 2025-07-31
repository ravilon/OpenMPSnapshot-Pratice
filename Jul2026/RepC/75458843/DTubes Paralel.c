#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define B 50
#define K 7

struct timeval start, end;
double timeclock;
int i,j,k;

//Step 1: Baca txt File
void BacaFile(FILE *file, char* matrix[B][K]){
    gettimeofday(&start, NULL);
    printf("Reading File... \n");
    char word[20];
    if (!file){
        printf("File tidak ditemukan"); //Cek File ada atau tidak (perlukah?)
    }

#pragma omp parallel num_threads(7)
{
    #pragma omp single nowait
    {
        if(!feof(file)){ //jika belum mencapai akhir file, maka
    #pragma omp task
    {
        for(i = 0; i < B; i++){
            for(j = 0; j < K-1; j++){
                fscanf(file,"%s",word); //baca file per maksimal 20 karakter
                matrix[i][j]=strdup(word); //Duplikasi dari word untuk dimasukkan ke dalam matrix
            }
        }
    }
        }
    }
}
    gettimeofday(&end, NULL);
    timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("Reading File Success \n");printf("Clock Time = %g \n",timeclock);
    printf("\n \n");
}

//Step 2: Parsing String ke Float
void ParseAndSave(char* dataX[B][K],float parse[B][K-2]){
    gettimeofday(&start, NULL);
    printf("Parsing Data from String to Number...\n");

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
    for(i=0;i<B;i++){
        //Parsing Case Index String
        parse[i][0]=atof(dataX[i][0]);
        //Parsing Half-Time String
        //This variable is based by days
        parse[i][1]=atof(dataX[i][3]);
        //Parsing Shut Down Year String
        //Converting from years to days
        parse[i][2]=atof(dataX[i][4]);
        parse[i][2]=(2016-parse[i][2])*365;
        //Parsing Mass when Reactor Shut Down String
        parse[i][3]=atof(dataX[i][5]);
    }
}

    //The Rest of Columns is Predict Result
    gettimeofday(&end, NULL);
    timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("Parsing Complete \n");printf("Clock Time = %g \n",timeclock);
    printf("\n \n");
}

//Step 3: Hitung Sisa Limbah pada tahun ini
void Predict(float parse[B][K-2]){
    gettimeofday(&start, NULL);
    printf("Calculating data to Predict Nuclear Waste Mass\n"); j=4;

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
        for(i=0;i<B;i++){
            parse[i][j]=parse[i][3]*(pow(0.5,(parse[i][2]/parse[i][1])));
        }
}

    gettimeofday(&end, NULL);
    timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("Calculation Complete! \n");printf("Clock Time = %g \n",timeclock);
    printf("\n \n");
}

//Step 4: Print semua pabrik yang masih ada sisa limbahnya pada tahun ini
void FindNotZero(float parse[B][K-2],char* dataX[B][K]){
    gettimeofday(&start, NULL); int th_id;
    printf("Print Ex-Nuclear Plant that still have Nuclear Waste \n");

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
    for(i=0;i<B;i++){
        if(parse[i][4]!=0){
        th_id=omp_get_thread_num();
        printf(" Thread Id: %d ",th_id);
        printf(" Nomor: %s\n Nama Pabrik: %s\n Bentuk Limbah: %s\tWaktu Paruh Isotop: %s Hari\t Tahun ditutup: %s\t Massa Limbah pada Tahun Tersebut: %s Pound\n Massa Limbah pada saat ini: %10.60f Pound\n \n",dataX[i][0],dataX[i][1],dataX[i][2],dataX[i][3],dataX[i][4],dataX[i][5],parse[i][4]);
        }
    }
}

    gettimeofday(&end, NULL);
    timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("Clock Time = %g \n",timeclock);
}

//Main Program
int main(){
	FILE *baca;
	char* data[B][K]; //Matrix yang menyimpan data dari txt File
	float parsed[B][K-2];
	float wp,cy,massa; //wp = waktu paruh, cy = close year, massa = jumlah limbah awal

	gettimeofday(&start, NULL);

	baca=fopen("database 5000.txt","r");

    BacaFile(baca,data);
    fclose(baca);
    //isi file telah dibaca dan seluruh matriks telah terisi
    //saatnya menutup file

    ParseAndSave(data,parsed);
    //data berbentuk nomor sudah diganti ke tipe data float

    Predict(parsed);
    //Hitung data untuk memperkirakan sisa limbah pada setiap pabrik

    FindNotZero(parsed,data);
    //Temukan Pabrik mana saja yang limbah nuklirnya masih ada

    gettimeofday(&end, NULL);
    timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("\nClock Time of Entire Process = %g \n",timeclock);
	return 0;
}
