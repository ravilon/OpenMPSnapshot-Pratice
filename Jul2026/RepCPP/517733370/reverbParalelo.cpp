//*------------------
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "portaudio.h"
#include <sndfile.h>
#include <chrono>
#define NUM_SECONDS (10000)
#define SAMPLE_RATE (44100) //44.1KHz
#define FRAMES_PER_BUFFER (1024)
#define DELAY_SIZE (SAMPLE_RATE*2)

//Arreglos para almacenar el procesado de los combFilters
float* combF1=new float[FRAMES_PER_BUFFER];
float* combF2=new float[FRAMES_PER_BUFFER];
float* combF3=new float[FRAMES_PER_BUFFER];
float* combF4=new float[FRAMES_PER_BUFFER];
//Arreglo de salida de CombFilters
float* OutCombF=new float[FRAMES_PER_BUFFER];
//Arreglos de salida de AllPassFilters
float* AllPF1=new float[FRAMES_PER_BUFFER];
float* AllPF2= new float[FRAMES_PER_BUFFER];
//medida de tiempo
auto start=std::chrono::high_resolution_clock::now();
auto end=std::chrono::high_resolution_clock::now();
auto tiempo=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
double total=0;
int counter=0;
typedef struct{
int posActual;
float monoDelay[DELAY_SIZE];
int mono_delayedSamplePos;
} paAudioData;

/*Conceptos:
* FRAMES_PER_BUFFER= tamaño del buffer de entrada o cantidad de muestras de audio.
* SAMPLE_RATE= frecuencia con la que se toman muestras de sonido.
* DELAY_SIZE = tamaño del buffer que almacena las muestras para luego aplicar en las funciones que utilizan retraso de onda.
* Señal de entrada corresponde a un número float
que representa la amplitud de la onda de sonido (conversión PCM).
* delay= retraso en milisegundos de la señal.
* decay= reducción de la amplitud de la señal.
*/
//combFilter
void combFilter(float* in,float* comb, float decay, paAudioData* data){
int posActual, monoDelayedPos;
posActual= data->posActual;
monoDelayedPos= posActual - data->mono_delayedSamplePos;
while(monoDelayedPos < 0){
monoDelayedPos+=DELAY_SIZE;
}
//MODO NAIVE
#pragma omp parallel for shared(in, data, posActual)
for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
data->monoDelay[(posActual + i) % DELAY_SIZE] = in[i];
}
//y[n] = x[n] + decay*y[n-delay]
#pragma omp parallel for shared(posActual,monoDelayedPos,decay, data, comb)
for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
//std::cout << i << " " << omp_get_thread_num << std::endl;
comb[i] = (data->monoDelay[(posActual + i) % DELAY_SIZE] + data->monoDelay[(monoDelayedPos + i) % DELAY_SIZE] * decay);
}
/*
//dividir por cuatro para hacerlo mas "secuencial"
//primer cuarto
#pragma omp parallel for shared(in, data, posActual)
for (int i = 0; i < FRAMES_PER_BUFFER/4; i++) {
data->monoDelay[(posActual + i) % DELAY_SIZE] = in[i];
}
#pragma omp parallel for shared(posActual,monoDelayedPos,decay, data, comb)
for (int i = 0; i < FRAMES_PER_BUFFER/4; i++) {
//std::cout << i << " " << omp_get_thread_num << std::endl;
comb[i] = (data->monoDelay[(posActual + i) % DELAY_SIZE] + data->monoDelay[(monoDelayedPos + i) % DELAY_SIZE] * decay);
}
//segundo cuarto
#pragma omp parallel for shared(in, data, posActual)
for (int i = FRAMES_PER_BUFFER/4; i < FRAMES_PER_BUFFER/2; i++) {
data->monoDelay[(posActual + i) % DELAY_SIZE] = in[i];
}
#pragma omp parallel for shared(posActual,monoDelayedPos,decay, data, comb)
for (int i = FRAMES_PER_BUFFER/4; i < FRAMES_PER_BUFFER/2; i++) {
//std::cout << i << " " << omp_get_thread_num << std::endl;
comb[i] = (data->monoDelay[(posActual + i) % DELAY_SIZE] + data->monoDelay[(monoDelayedPos + i) % DELAY_SIZE] * decay);
}
//tercer cuarto
#pragma omp parallel for shared(in, data, posActual)
for (int i = FRAMES_PER_BUFFER/2; i < 3*FRAMES_PER_BUFFER/4; i++) {
data->monoDelay[(posActual + i) % DELAY_SIZE] = in[i];
}
#pragma omp parallel for shared(posActual,monoDelayedPos,decay, data, comb)
for (int i = FRAMES_PER_BUFFER/2; i < 3*FRAMES_PER_BUFFER/4; i++) {
//std::cout << i << " " << omp_get_thread_num << std::endl;
comb[i] = (data->monoDelay[(posActual + i) % DELAY_SIZE] + data->monoDelay[(monoDelayedPos + i) % DELAY_SIZE] * decay);
}
//cuarto cuarto
#pragma omp parallel for shared(in, data, posActual)
for (int i = 3*FRAMES_PER_BUFFER/4; i < FRAMES_PER_BUFFER; i++) {
data->monoDelay[(posActual + i) % DELAY_SIZE] = in[i];
}
#pragma omp parallel for shared(posActual,monoDelayedPos,decay, data, comb)
for (int i = 3*FRAMES_PER_BUFFER/4; i < FRAMES_PER_BUFFER; i++) {
//std::cout << i << " " << omp_get_thread_num << std::endl;
comb[i] = (data->monoDelay[(posActual + i) % DELAY_SIZE] + data->monoDelay[(monoDelayedPos + i) % DELAY_SIZE] * decay);
}
*/
data->posActual=(posActual+FRAMES_PER_BUFFER)%DELAY_SIZE;
}
void allPassFilter(float* apIn, float* apOut,float decay,paAudioData* data){
int posActual, monoDelayedPos;
float* apOutStart=apOut;
posActual=data->posActual;
monoDelayedPos= posActual - data->mono_delayedSamplePos;
while(monoDelayedPos<=0){
monoDelayedPos+=DELAY_SIZE;
}
//MODO NAIVE
#pragma omp parallel for shared(posActual, apIn, data)
for(int i=0;i<FRAMES_PER_BUFFER;i++){
data->monoDelay[(posActual+i)%DELAY_SIZE]=apIn[i];
}
//y[n]= -g*x[n] + x[n-delay] + g*y[n-delay]
#pragma omp parallel for shared(data, posActual, monoDelayedPos, decay,apOut)
for(int i=0;i<FRAMES_PER_BUFFER;i++){
apOut[i] =(-decay*data->monoDelay[(posActual+i)%DELAY_SIZE])+
(data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE])+
(decay*data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE]);
}

/*
//DIVIDIR POR 4 PARA HACERLO MAS "SECUENCIAL"
//primer cuarto
#pragma omp parallel for shared(posActual, apIn, data)
for(int i=0;i<FRAMES_PER_BUFFER/4;i++){
data->monoDelay[(posActual+i)%DELAY_SIZE]=apIn[i];
}
#pragma omp parallel for shared(data, posActual, monoDelayedPos, decay,apOut)
for(int i=0;i<FRAMES_PER_BUFFER/4;i++){
apOut[i] =(-decay*data->monoDelay[(posActual+i)%DELAY_SIZE])+(data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE])+(decay*data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE]);
}
//segundo cuarto
#pragma omp parallel for shared(posActual, apIn, data)
for(int i=FRAMES_PER_BUFFER/4;i<FRAMES_PER_BUFFER/2;i++){
data->monoDelay[(posActual+i)%DELAY_SIZE]=apIn[i];
}
#pragma omp parallel for shared(data, posActual, monoDelayedPos, decay,apOut)
for(int i=FRAMES_PER_BUFFER/4;i<FRAMES_PER_BUFFER/2;i++){
apOut[i] =(-decay*data->monoDelay[(posActual+i)%DELAY_SIZE])+(data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE])+(decay*data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE]);
}
//tercer cuarto
#pragma omp parallel for shared(posActual, apIn, data)
for(int i=FRAMES_PER_BUFFER/2;i<3*FRAMES_PER_BUFFER/4;i++){
data->monoDelay[(posActual+i)%DELAY_SIZE]=apIn[i];
}
#pragma omp parallel for shared(data, posActual, monoDelayedPos, decay,apOut)
for(int i=FRAMES_PER_BUFFER/2;i<3*FRAMES_PER_BUFFER/4;i++){
apOut[i] =(-decay*data->monoDelay[(posActual+i)%DELAY_SIZE])+(data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE])+(decay*data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE]);
}
//cuarto cuarto
#pragma omp parallel for shared(posActual, apIn, data)
for(int i=3*FRAMES_PER_BUFFER/4;i<FRAMES_PER_BUFFER;i++){
data->monoDelay[(posActual+i)%DELAY_SIZE]=apIn[i];
}
#pragma omp parallel for shared(data, posActual, monoDelayedPos, decay,apOut)
for(int i=3*FRAMES_PER_BUFFER/4;i<FRAMES_PER_BUFFER;i++){
apOut[i] =(-decay*data->monoDelay[(posActual+i)%DELAY_SIZE])+(data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE])+(decay*data->monoDelay[(monoDelayedPos+i)%DELAY_SIZE]);
}
*/
data->posActual=(posActual+FRAMES_PER_BUFFER)%DELAY_SIZE;
//Normalización de la señal para evitar distorsión
float val= *apOut;
float valAct;
float max=0.0f;//valor máximo de la señal
for(int i=0;i<FRAMES_PER_BUFFER;i++){
if(fabs(apOut[i])>max){
max=fabs(apOut[i]);
}
}
//std::cout<<max<<std::endl;
if(max==0.0f){
max=1.05f;//en caso de que el máximo sea 0 para evitar dividir por 0
}
#pragma omp parallel for private(valAct, val) shared(max, apOut)
for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
valAct = apOut[i];
val = (val + (valAct - val)) / max;
apOut[i] = val;
}
}

SNDFILE* file;
void reverb(float* in, float* out, paAudioData* data, float decay, float mix){
float* posInOut=out;
//Combfilters
#pragma omp parallel
{
#pragma omp taskgroup
{
#pragma omp task shared(in, decay, combF1, data)
combFilter(in, combF1, decay, &data[0]);
#pragma omp task shared(in, decay, combF2, data)
combFilter(in, combF2, decay - 0.131f, &data[1]);
#pragma omp task shared(in, decay, combF3, data)
combFilter(in, combF3, decay - 0.154f, &data[2]);
#pragma omp task shared(in, decay, combF4, data)
combFilter(in, combF4, decay - 0.11f, &data[3]);
}
}
#pragma omp parallel for shared(OutCombF, combF1, combF2, combF3, combF4)
for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
//std::cout << i << " " << omp_get_thread_num << omp_get_num_threads<< std::endl;
OutCombF[i] = combF1[i] + combF2[i] + combF3[i] + combF4[i];
}
//AllPassFilters
allPassFilter(OutCombF, AllPF1, 0.7f, &data[4]);//AllPassFilter1
allPassFilter(AllPF1, AllPF2, 0.7f, &data[5]);//AllPassFilter2
allPassFilter(AllPF2, out, 0.7f, &data[6]);//AllPassFilter3
// Wet/Dry Mix
#pragma omp parallel for shared(in, out, mix)
for(int i=0;i<FRAMES_PER_BUFFER;i++) {
//Dry= input sin procesar, Wet= input procesado
out[i] = (in[i]*(0.1f-mix)) + (out[i]*mix);
}
sf_write_float(file,posInOut,FRAMES_PER_BUFFER);
}
//Función que recibe la señal de entrada y envia la señal procesada
static int reverbCallback(const void *inputBuffer,
void *outputBuffer,
unsigned long framesPerBuffer,
const PaStreamCallbackTimeInfo* timeInfo,
PaStreamCallbackFlags statusFlags,
void *userData){
paAudioData* data =(paAudioData*) userData;
float* in=(float*)inputBuffer;
float* out=(float*)outputBuffer;
start=std::chrono::high_resolution_clock::now();
reverb(in,out,data,0.837f,0.1f);
end=std::chrono::high_resolution_clock::now();
tiempo=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
total+=tiempo;
counter+=1;
return 0;
}
static paAudioData data[7];
void dataInit(){
for(int j=0;j<7;j++){
data[j].posActual=0;
for(int i=0;i<DELAY_SIZE;i++){
data[j].monoDelay[i]=0.0f;
}
}
data[0].mono_delayedSamplePos=SAMPLE_RATE*0.501;
data[1].mono_delayedSamplePos=SAMPLE_RATE*0.678;
data[2].mono_delayedSamplePos=SAMPLE_RATE*0.511;
data[3].mono_delayedSamplePos=SAMPLE_RATE*0.323;
data[4].mono_delayedSamplePos=SAMPLE_RATE*0.125;
data[5].mono_delayedSamplePos=SAMPLE_RATE*0.042;
data[6].mono_delayedSamplePos=SAMPLE_RATE*0.012;
}
static void create_file(const char* fname, int format){
SF_INFO sfinfo;
sfinfo.channels=1;
sfinfo.samplerate=SAMPLE_RATE;
sfinfo.format=format;
std::cout<<"Creando archivo reverbPar.wav\nFormato: "<<format<<std::endl;
file=sf_open(fname,SFM_WRITE,&sfinfo);
}
int main(void){
int numDevices, defaultDisplayed;
const PaDeviceInfo *deviceInfo;
const char* fname="/home/surt/CLionProjects/untitled/reverbPar.wav";
create_file(fname,SF_FORMAT_WAV | SF_FORMAT_PCM_16);
PaStream *stream;
PaError err;
err=Pa_Initialize();
if(err!=paNoError) goto error;
numDevices=Pa_GetDeviceCount();
if(numDevices<0){
std::cout<<"Error: Pa_countDevices returned "<<numDevices<<std::endl;
err=numDevices;
goto error;
}
/*
//PARAMETROS PARA USAR EL MIXER COMO ENTRADA
PaStreamParameters inParams;
inParams.device=6;
inParams.channelCount=1;
inParams.sampleFormat=paFloat32;
inParams.suggestedLatency= Pa_GetDeviceInfo(6)->defaultLowInputLatency;
inParams.hostApiSpecificStreamInfo=NULL;
//PARAMETROS PARA USAR EL MIXER COMO SALIDA
//*
PaStreamParameters outParams;
outParams.device=6;
outParams.channelCount=1;
outParams.sampleFormat=paFloat32;
outParams.suggestedLatency= Pa_GetDeviceInfo(6)->defaultLowOutputLatency;
outParams.hostApiSpecificStreamInfo=NULL;
*/
//INICIALIZA LAS ESTRUCTURAS CON LOS BUFFERS DEL REVERB
dataInit();
//IMPRIME LOS DISPOSITIVOS DISPONIBLES EN EL COMPUTADOR
//EN MI CASO, EL MIXER ES EL DISPOSITIVO NUMERO "6", O SEA, EL SEPTIMO YA QUE CUENTA DESDE EL 0
for(int i=0;i<numDevices;i++){
deviceInfo= Pa_GetDeviceInfo(i);
deviceInfo = Pa_GetDeviceInfo( i );
std::cout<<"Device No: "<< i <<std::endl;
defaultDisplayed = 0;
if( i == Pa_GetDefaultInputDevice()){
printf( "[ Default Input" );
defaultDisplayed = 1;
}
else if( i == Pa_GetHostApiInfo( deviceInfo->hostApi )->defaultInputDevice){
const PaHostApiInfo *hostInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
std::cout<< "[ Default "<< hostInfo->name<<" Input"<<std::endl;
defaultDisplayed = 1;
}
if( i == Pa_GetDefaultOutputDevice()){
std::cout<< (defaultDisplayed ? "," : "[");
std::cout<< " Default Output";
defaultDisplayed = 1;
}
else if( i == Pa_GetHostApiInfo( deviceInfo->hostApi )->defaultOutputDevice){
const PaHostApiInfo *hostInfo = Pa_GetHostApiInfo(deviceInfo->hostApi);
std::cout<< (defaultDisplayed ? "," : "[") <<std::endl;
std::cout<< " Default "<<hostInfo->name<<" Output" <<std::endl;
defaultDisplayed = 1;
}
if(defaultDisplayed){
std::cout<<" ]"<<std::endl;
}
std::cout<< "Name                        = "<< deviceInfo->name <<std::endl;
std::cout<< "Host API                    = "<<  Pa_GetHostApiInfo( deviceInfo->hostApi )->name <<std::endl;
std::cout<< "Max inputs                  = "<< deviceInfo->maxInputChannels  <<std::endl;
std::cout<< "Max outputs                 = "<< deviceInfo->maxOutputChannels  <<std::endl;
std::cout<< "Default low input latency   = "<<deviceInfo->defaultLowInputLatency  <<std::endl;
std::cout<< "Default low output latency  = "<<deviceInfo->defaultLowOutputLatency  <<std::endl;
std::cout<< "Default high input latency  = "<<deviceInfo->defaultHighInputLatency  <<std::endl;
std::cout<< "Default high output latency = "<<deviceInfo->defaultHighOutputLatency  <<std::endl;
std::cout<<"\n";
}
//*/
//ABRE UN STREAM EN EL DISPOSITIVO POR DEFECTO
err= Pa_OpenDefaultStream(&stream,// input stream//
1,//n° input channels//
1,//n° output channels//
paFloat32, //32 bit float output//
SAMPLE_RATE, //44.1KHz //
FRAMES_PER_BUFFER,
reverbCallback, //CallbackFx
&data);
/*
//ABRE UN STREAM PARA EL MIXER
err= Pa_OpenStream(&stream,
&inParams,
&outParams,
SAMPLE_RATE,
FRAMES_PER_BUFFER,
0,
reverbCallback,
&data);
*/
if(err!= paNoError) goto error;
err= Pa_StartStream(stream);
if(err!= paNoError) goto error;
Pa_Sleep(NUM_SECONDS);
std::cout<<"Tiempo promedio de procesado para: "<<SAMPLE_RATE<<" buffer: "<<FRAMES_PER_BUFFER<<"\n"<<total/(double)counter<<" ns"<<std::endl;
err= Pa_StopStream(stream);
sf_close(file);
if(err!= paNoError) goto error;
err= Pa_CloseStream(stream);
if(err!= paNoError) goto error;
Pa_Terminate();
std::cout<<"Yay!"<<std::endl;
delete[] combF1;
delete[] combF2;
delete[] combF3;
delete[] combF4;
delete[] OutCombF;
delete[] AllPF1;
delete[] AllPF2;
return err;
error:
Pa_Terminate();
std::cout<<"Error: "<<Pa_GetErrorText(err)<<std::endl;
return err;
}