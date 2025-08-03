
#pragma warning(disable:4996)
//This pragma can be used to suppress the warnings for fopen and fscanf without editing preprocessor definitions
#include <Windows.h>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;


/*-----Structure Definitions and read/write  .ppm functions from Project One-----*/
//in our model sample image 
//PPM header is : 
//P6
//2452 2132 //Number of columns(width) Number of rows (Height)
//255  //Max channel value
typedef struct {
unsigned char red, green, blue;
} PPMPixel;

typedef struct {
int x, y;
PPMPixel *data;
} PPMImage;

constexpr auto RGB_COMPONENT_COLOR = 255;

//Define the number of iterations
const int iter = 100;
//const int threshold = 255;


static PPMImage *readPPM(const char *filename)
{
char buff[16];
PPMImage *img = 0;
FILE *fp;
//ifstream inputFile(filename, ios::in | ios::binary);
int c, rgb_comp_color;
//open PPM file for reading
fp = fopen(filename, "rb");//reading a binary file
if (!fp) {
fprintf(stderr, "Unable to open file '%s'\n", filename);
exit(1);
}


//read image format
if (!fgets(buff, sizeof(buff), fp)) {
perror(filename);
exit(1);
}

//check the image format can be P3 or P6. P3:data is in ASCII format P6: data is in byte format
if (buff[0] != 'P' || buff[1] != '6') {
fprintf(stderr, "Invalid image format (must be 'P6')\n");
exit(1);
}

//alloc memory form image
img = (PPMImage *)malloc(sizeof(PPMImage));
if (!img) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

//check for comments
c = getc(fp);
while (c == '#') {
while (getc(fp) != '\n');
c = getc(fp);
}

ungetc(c, fp);//last character read was out back
//read image size information
if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {//if not reading widht and height of image means if there is no 2 values
fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
exit(1);
}

//read rgb component
if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
exit(1);
}

//check rgb component depth
if (rgb_comp_color != RGB_COMPONENT_COLOR) {
fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
exit(1);
}

while (fgetc(fp) != '\n');
//memory allocation for pixel data for all pixels
img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

if (!img) {
fprintf(stderr, "Unable to allocate memory\n");
exit(1);
}

//read pixel data from file
if (fread(img->data, 3 * img->x, img->y, fp) != img->y) { //3 channels, RGB  //size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
/*ptr − This is the pointer to a block of memory with a minimum size of size*nmemb bytes.
size − This is the size in bytes of each element to be read.
nmemb − This is the number of elements, each one with a size of size bytes.
stream − This is the pointer to a FILE object that specifies an input stream.
*/
fprintf(stderr, "Error loading image '%s'\n", filename);
exit(1);
}

fclose(fp);
return img;
}



void writePPM(const char *filename, PPMImage *img)
{
FILE *fp;
//open file for output
fp = fopen(filename, "wb");//writing in binary format
if (!fp) {
fprintf(stderr, "Unable to open file '%s'\n", filename);
exit(1);
}

//write the header file
//image format
fprintf(fp, "P6\n");

//image size
fprintf(fp, "%d %d\n", img->x, img->y);

// rgb component depth
fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

// pixel data
fwrite(img->data, 3 * img->x, img->y, fp);
fclose(fp);
}

/*---------------------------------------------------------------------------*/

/*-----Implementation of Conway's Game of Life using OpenMP*/

//Game Conditions
//Get the number of live neighbors for the input pixel
int getNumLiveNeighbors(int cellMatrix[][3]) {
int count = 0;
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
if (cellMatrix[i][j] > 0)
count++;
}
}
return count;
}
//Get the number of dead neighbors for the input pixel
int getNumDeadNeighbors(int cellMatrix[][3]) {
int count = 0;
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
if (cellMatrix[i][j] == 0)
count++;
}
}
return count;
}

void runGame(PPMImage *inputImage) {
int numLiveCells = 0;
int numDeadCells = 0;
char pixel = 0;
//Set rows and columns to the dimensions of the input image
int numCol = inputImage->x;
int numRow = inputImage->y;
int width = inputImage->x;
int height = inputImage->y;
int size = width * height;
int i = 0;
int j = 0;
int N = 0, NE = 0, NW = 0, E = 0, W = 0, S = 0, SE = 0, SW = 0;
int cellMatrix[3][3];
for (i = 0; i < (inputImage->x) * (inputImage->y); i++) {
//for (i = 0; i < Row; i++) {
//for (j = 0; j < Col; j++) {
//We can use a value of zero (black) or try a threshold value
if (((int)inputImage->data[i].red > 0  ) && ((int)inputImage->data[i].green > 0) && ((int)inputImage->data[i].blue > 0)) {

//Get the neighborhood values for a target live cell.
E = (i + 1);
W = (i - 1);
N = (i - width);
S = (i + width);
NE = (i - width + 1);
NW = (i - width - 1);
SW = (i + width - 1);
SE = (i + width + 1);

//Populate the neighborhood matrix
if ((NW >= 0) && ((i % width) != 0))
cellMatrix[0][0] = inputImage->data[NW].red;
if (N >= 0)
cellMatrix[0][1] = inputImage->data[N].red;
if (NE >= 0 && ((E % width) != 0))
cellMatrix[0][2] = inputImage->data[NE].red;
if (i % width != 0)
cellMatrix[1][0] = inputImage->data[W].red;
cellMatrix[1][1] = inputImage->data[i].red;
if (E % width != 0)
cellMatrix[1][2] = inputImage->data[E].red;
if ((SW < size) && ((i % width) != 0))
cellMatrix[2][0] = inputImage->data[SW].red;
if ((S < size))
cellMatrix[2][1] = inputImage->data[S].red;
if ((SE < size) && ((E % width) != 0))
cellMatrix[2][2] = inputImage->data[SE].red;

numLiveCells = getNumLiveNeighbors(cellMatrix);
numDeadCells = getNumDeadNeighbors(cellMatrix);
//For Spaces that are populated:
//Case One: populated cell with one or no live neighbors dies
if (numLiveCells == 1 || numLiveCells == 2) {
inputImage->data[i].red = 0;
inputImage->data[i].green = 0;
inputImage->data[i].blue = 0;
}
//Case Two: populated cell with four or more live neighbors dies
if (numLiveCells >= 4) {
inputImage->data[i].red = 0;
inputImage->data[i].green = 0;
inputImage->data[i].blue = 0;
}
//No operations for Case Three (populated cell with two or three live neighbors survives)

//-----Debugging: Check that live cells and neighbors exist-----//
//cout << "Live cell found in cell: " << i << endl;
//cout << "Number of live cells: " << numLiveCells <<endl;
//cout << "Number of dead cells: " << numDeadCells << endl;
//	if(cellMatrix[0][0] == 255)
//	cout << "Live NW neighbor at " << NW << "\n";
//	if (cellMatrix[0][1] == 255)
//		cout << "Live N neighbor at " << N << "\n";
//	if(cellMatrix[0][2] == 255)
//		cout << "Live NE neighbor at " << NE << "\n";
//	if (cellMatrix[1][0] == 255)
//		cout << "Live W neighbor at " << W << "\n";
//if (cellMatrix[1][2] == 255 && cellMatrix[1][0] == 255 && cellMatrix[0][1] == 255 && cellMatrix[0][2] ==255) {
//	cout << "Live cell found in cell: " << i << endl;
//	cout << "Live E neighbor at " << E << "\n";
//	cout << "Live N neighbor at " << N << "\n";
//	cout << "Live W neighbor at " << W << "\n";
//	cout << "Live NW neighbor at " << NW << "\n";
//	cout << "Number of live cells: " << numLiveCells << endl;
//	cout << "Number of dead cells: " << numDeadCells << endl;
//}

}
else if (((int)inputImage->data[i].red < 100) && ((int)inputImage->data[i].green < 100) && ((int)inputImage->data[i].blue < 100)) {
//Get the neighborhood values for a target dead cell.
E = (i + 1);
W = (i - 1);
N = (i - width);
S = (i + width);
NE = (i - width + 1);
NW = (i - width - 1);
SW = (i + width - 1);
SE = (i + width + 1);

//Populate the neighborhood matrix
if ( (NW >= 0) && ( (i % width) != 0))
cellMatrix[0][0] = inputImage->data[NW].red;
if(N >= 0)
cellMatrix[0][1] = inputImage->data[N].red;
if(NE >=0 && ((E % width)!=0))
cellMatrix[0][2] = inputImage->data[NE].red;
if(i % width != 0)
cellMatrix[1][0] = inputImage->data[W].red;
cellMatrix[1][1] = inputImage->data[i].red;
if( E % width !=0)
cellMatrix[1][2] = inputImage->data[E].red;
if( (SW <size) && ((i % width) !=0) )
cellMatrix[2][0] = inputImage->data[SW].red;
if( (S < size))
cellMatrix[2][1] = inputImage->data[S].red;
if((SE < size)&& ((E % width) !=0))
cellMatrix[2][2] = inputImage->data[SE].red;

numLiveCells = getNumLiveNeighbors(cellMatrix);
numDeadCells = getNumDeadNeighbors(cellMatrix);

//Cells come to life when number of live neighbors is three
if (numLiveCells == 3) {
inputImage->data[i].red = 255;
inputImage->data[i].green = 255;
inputImage->data[i].blue = 255;
}
}
}
}



int main() {

PPMImage *inputImage, *outputImage;
string temp = "";
int i = 0;
cout << "Maximum number of threads:  " << omp_get_max_threads() << endl;
cout << "Reading Edge-Detected model.ppm file...\n";
auto start = high_resolution_clock::now();
inputImage = readPPM("sobel.ppm");
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << "Finished in " << duration.count() << " microseconds" << endl;
outputImage = inputImage;

cout << "Running Conway's Game of Life...\n";
cout << "Serial...\n";

start = high_resolution_clock::now();

for (int i = 0; i < iter; i++) {
inputImage = outputImage;
runGame(inputImage);
temp = "Iteration" + to_string(i + 1) + ".ppm";
//Convert temp string to char* and write to file
writePPM(temp.c_str(), outputImage);
}

stop = high_resolution_clock::now();
auto duration2 = duration_cast<seconds>(stop - start);
cout << "Finished in " << duration2.count() << " seconds" << endl;


cout << "Parallel...\n";
start = high_resolution_clock::now();
#pragma omp parallel for
for ( int i = 0; i < iter; i++) {
inputImage = outputImage;
runGame(inputImage);
temp = "Iteration" + to_string(i + 1) + ".ppm";
//Convert temp string to char* and write to file
writePPM(temp.c_str(), outputImage);
}
stop = high_resolution_clock::now();
auto duration3 = duration_cast<seconds>(stop - start);
cout << "Finished in " << duration3.count() << " seconds" << endl;

//Serial/Parallel time
auto speedup = (duration2.count() / duration3.count());
cout << "Speedup is " << speedup;

return 0;
}


