//comando para ejecutar
// clear; gcc cracker.c -o cracker; ./cracker 2 2 numeric

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "md5_hasher.h"

//#define DEBUG

#define SUCCESS_CODE 0
#define ERROR_CODE 2
#define NOT_CHARSET_VALID 127

#define MULTIPLE_LEN

#define NOT_CRACKED 0
#define PAWNED 1

#define KEEP_RUNNING 0
#define STOP 1

//global constants
static const char numeric[] = "0123456789";
static const char numeric_space[] = "0123456789 ";
static const char alpha[] = "abcdefghijklmnopqrstuvwxyz";
static const char alpha_space[] = "abcdefghijklmnopqrstuvwxyz ";

void showme(){
      printf("[Thread %d of %d]\n", omp_get_thread_num(), omp_get_num_threads());
}
int ipow(int source, int exp)
{
  long base = source;
  long result = 1.0L;
  while (exp)
  {
    if (exp & 1)
    result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

char* getKeyIndex(int key_index, int key_size, const char *charset)
{
  #ifdef DEBUG
  printf("Retrieving key %d of size %d\n", key_index, key_size);
  #endif
  //get memory for key
  char *key;
  key = (char*) malloc( sizeof(char) * (key_size) );

  //get charset length
  const int charset_length = strlen(charset);

  //calculate a key given an index
  int divisionNumber;
  for(divisionNumber = 0; divisionNumber < key_size; divisionNumber++){
    //showme();
    key[key_size-divisionNumber-1] = charset[key_index%charset_length];
    key_index = key_index/charset_length;
  }
  return key;
}

void printKey(char *key, int password_length, int charset_length)
{
  int i;
  for(i = 0; i < password_length; i++){
    printf("%c", key[i]);
  }
  printf("\n");
}

#ifdef MULTIPLE_LEN
int estimateLength(int value, int min, int max, int charset_length){
  int current_key_length = min;
  int found = 0;
  do{
    long key_space = ipow(charset_length, current_key_length);
    found = value < key_space;
    current_key_length++;
  }while(current_key_length <= max && !found);
  return current_key_length-1;
}
#endif

int execute(int start_value, int min, int max, char* target, const char* charset)
{
  //generated from globals
  int charset_length = strlen(charset);

  //calculate key space
  long key_space = 0L;

  #ifdef MULTIPLE_LEN
  int i=0;
  #pragma omp parallel for reduction(+:key_space)
  for(i = min; i <= max; i++){
    long value = ipow(charset_length, i);
    key_space += value;
    #ifdef DEBUG
    showme();
    printf("\t\tSingle key space for length %d is:\t%ld\n", i, value);
    #endif
  }
  #else
  key_space = ipow(charset_length, max);
  #endif
  printf("\t\tBruteforce key space:\t%ld\n", key_space);
  //current password length
  int password_length;

  //key index
  int iterate;

  //generate keyspace
  int last_passw_len = min;
  int payload = 0;

  //temp variable to hold result
  char* found;

  //password cracked detection condition. per thread
  int password_cracked = NOT_CRACKED;
  long attempts = 0;
  //stop condition. multithread shared
  int stop_condition = 0;

  //start cracking until found
  #pragma omp parallel for shared(attempts,stop_condition)
  for(iterate = start_value ; iterate < key_space; iterate++){
    if(stop_condition == KEEP_RUNNING) {
      #pragma omp critical
      {
	      #ifdef DEBUG
	      printf("[Thread %d of %d] - %d %d \n", omp_get_thread_num(), omp_get_num_threads(), iterate, stop_condition);
	      #endif
	      int idx = iterate;
	      #ifdef MULTIPLE_LEN
	      //password length will be variable depending on given key and min max values.
	      int password_length = estimateLength(idx, min, max, charset_length);
	      if(password_length != last_passw_len){
		payload = ipow(charset_length, last_passw_len);
		idx = iterate - payload;
	      }
	      #endif
	      //get key given an index
	      char *key = getKeyIndex(idx, password_length, charset);
	      char* hash = compute(key, password_length);
	      #ifdef DEBUG
	      printf("[Thread %d of %d] - Iteration: %d Key: %s MD5: %s\n", omp_get_thread_num(), omp_get_num_threads(), iterate, key, hash);
	      #endif
	      //compare it
	      if(strcmp(hash, target)==0){
		found = key;
		#ifdef DEBUG
		printf("[Thread %d of %d] - FOUND THE PASSWD: %d Key: %s MD5: %s\n", omp_get_thread_num(), omp_get_num_threads(), iterate, key, hash);
		#endif
		//release hash
		free(hash);
		hash = NULL;
		//pass cracked. stop
		password_cracked = PAWNED;
		stop_condition = STOP;
	      }
              //increase attempt count
	      attempts = attempts + 1;
	      }
	      //check if current thread found passwd or any other thread found it
	      if(password_cracked == PAWNED || stop_condition == STOP){
		stop_condition = STOP;
	      }
      }
      {
    }
  } //end for

  //show result
  if(password_cracked == PAWNED){
    printf("\n\n\t#################\n");
    printf("\tPassword cracked:\n");
    printf("\t#################\n\n" );
    printf("\n\tAttepts: %li \t of \t %li\n", attempts, key_space);
    printf("\tPassword is: %s\n\n", found);
    //release found key
    free(found);
    found = NULL;
  }
  else{
    printf("\n\n\tNO HASH FOUND. SORRY :(\n\n");
  }
  return SUCCESS_CODE;
}

int brute_force(int start_value, char* min, char* max, char* hash, char* charset_name)
{
  //convert los char* to int
  int max_len = atoi(max);
  int min_len = atoi(min);

  if(min_len > max_len){
    printf("\tPlease specify a min value smaller or equal than max value\n");
    return ERROR_CODE;
  }

  if(strcmp(charset_name, "numeric")==0){
    return execute(start_value, min_len, max_len,hash, numeric);
  }
  else if(strcmp(charset_name, "numeric_space")==0){
    return execute(start_value, min_len, max_len, hash, numeric_space);
  }
  else if(strcmp(charset_name, "alpha")==0){
    return execute(start_value, min_len, max_len, hash, alpha);
  }
  else if(strcmp(charset_name, "alpha_space")==0){
    return execute(start_value, min_len, max_len, hash, alpha_space);
  }
  return NOT_CHARSET_VALID;
}

int main(int argc, char **argv)
{
  //sustituir esto por getops
  char* execution_path = argv[0];
  printf("\texecution_path:\t%s\n", execution_path);
  char* min_size = argv[1];
  printf("\tmin_size:\t%s\n", min_size);
  char* max_size = argv[2];
  printf("\tmax_size:\t%s\n", max_size);
  char* cores = argv[3];
  printf("\tthreads:\t%s\n", cores);
  char* charset = argv[4];
  printf("\tcharset:\t%s\n", charset);
  char* hash = argv[5];
  printf("\ttarget:\t%s\n", hash);
  int start_value = 0;

  //set openmp threads
  int thread_num = atoi(cores);
  if(thread_num >= 1){
    omp_set_num_threads(thread_num);
    printf("\tDedicated threads to cracking:\t%d\n", thread_num);
  }
  return brute_force(start_value, min_size, max_size, hash, charset);
}
