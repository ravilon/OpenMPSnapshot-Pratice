/// A collection of useful functions with char arrays
/// extracted while making the long message attack

#include "util_char_arrays.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* return index of key if it is found, -1 otherwise*/
uint64_t linear_search(uint8_t *key, uint8_t *array, size_t array_len, size_t key_len)
{
  // return the index of element if it exits, otherwise 0
  // don't use this function to test if an element exists in an array!
  
  for (size_t i=0; i<array_len; ++i) {
    /* printf("i=%lu\n", i); */
    /* print_byte_txt("found:", &array[i*key_len], key_len); */
    /* print_byte_txt("key  :", key, key_len); */
    
    if ( 0 == memcmp(key, &array[i*key_len], key_len) ){
      return i;
    }
      
  }
  
  return 0; /* not found */
}


void* linear_search_ptr(uint8_t *key, uint8_t *array, size_t array_len, size_t key_len)
{
  for (size_t i=0; i<array_len; ++i) {
    if ( 0 == memcmp(key, &array[i*key_len], key_len) ){
      return &array[i*key_len];
    }
    
  }
  
  return NULL; /* not found */
}





void print_byte_array(uint8_t* array, size_t nbytes)
{
  for (size_t i=0; i<nbytes; ++i) 
    printf("0x%02x, ",  array[i]);
  puts("");
}



int cmp_arrays(char* array1, char* array2, size_t len){

  /// memcmp gives strange results
  for (size_t i = 0; i<len; ++i){
    if (array1[i] != array2[i])
      return 0;
  }
  
  return 1;
}


void human_format(char * target, uint64_t n) {
        if (n < 1000) {
                sprintf(target, "%lu" , n);
                return;
        }
        if (n < 1000000) {
                sprintf(target, "%.1fK", n / 1e3);
                return;
        }
        if (n < 1000000000) {
                sprintf(target, "%.1fM", n / 1e6);
                return;
        }
        if (n < 1000000000000ll) {
                sprintf(target, "%.1fG", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll) {
                sprintf(target, "%.1fT", n / 1e12);
                return;
        }
}


void print_char(unsigned char* l, size_t len){
  printf("0x");
  for (size_t i = 0; i<len; ++i)
    printf("%02x",(unsigned char) l[i]);
  puts("");
}

void print_u16(uint16_t* l, size_t len){
  printf("0x");
  for (size_t i = 0; i<len; ++i)
    printf("%02x", l[i]);
  puts("");
}


void print_byte_txt(char* txt, unsigned char* a, size_t len){
  printf("%s 0x", txt);
  for (size_t i = 0; i<len; ++i)
    printf("%02x",(unsigned char) a[i]);
  puts("");
}
  


/* unsigned char* long_message_zeros(size_t n_of_bits){ */
/*   /\*     DESCRIPTION        *\/ */
/*   /// Create array of zeros */
/*   /// input: n_of_bits: how many bits the output size should been */
/*   /// output: array of 0 bytes that has `n_of_bits` bits */
/*   /// e.g. input := 9, we can accommedate 9 bits in two bytes */
/*   size_t n_of_bytes = (size_t) ceil(n_of_bits/8.0); */
/*   unsigned char* A = (unsigned char *) malloc(sizeof(unsigned char) * n_of_bytes); */
  
/*   for (size_t i = 0; i<n_of_bytes; ++i) */
/*     A[i] = 0; */

/*   return A; */
/* } */


unsigned char* create_radom_byte_array(int n_of_bytes){
  /* Create seemingly a random byte array with total_n_of_bits */
  /// INPUT: how many bytes
  /// OUTPUT: array with ceil(total_n_of_bytes) entries
  ///         the last entry doesn't necessarily use all the 8 bits
  unsigned char* A = (unsigned char *)malloc(sizeof(unsigned char)*n_of_bytes);

  int d = 0;
  for (size_t i=0; i<n_of_bytes; ++i){
    d = 0 + rand() / (RAND_MAX / (255 - 0 + 1) + 1);
    A[i] = (unsigned char) d;
  }
  // int returned_bytes = getrandom(A, n_of_bytes, 1);
  // ++returned_bytes; // dummy operation to avoid not used warning

  return A;
}




inline void fill_radom_byte_array(unsigned char* A, int n_of_bytes, unsigned int *seed){
  /* Create seemingly a random byte array with total_n_of_bits */
  /// INPUT: how many bytes
  /// OUTPUT: array with ceil(total_n_of_bytes) entries
  ///         the last entry doesn't necessarily use all the 8 bits
  

  /* int d = 0; */
  /* for (size_t i=0; i<n_of_bytes; ++i){ */
  /*   d = 0 + rand_r(seed) / (RAND_MAX / (255 - 0 + 1) + 1); */
  /*   A[i] = (unsigned char) d; */
  /* } */
  int returned_bytes = getrandom(A, n_of_bytes, 1);
  ++returned_bytes; // dummy operation to avoid not used warning

}


void fill_radom_byte_array_get_random(unsigned char* A, int n_of_bytes){
  /* Create seemingly a random byte array with total_n_of_bits */
  /// INPUT: how many bytes
  /// OUTPUT: array with ceil(total_n_of_bytes) entries
  ///         the last entry doesn't necessarily use all the 8 bits
  

  /* int d = 0; */
  /* for (size_t i=0; i<n_of_bytes; ++i){ */
  /*   d = 0 + rand_r(seed) / (RAND_MAX / (255 - 0 + 1) + 1); */
  /*   A[i] = (unsigned char) d; */
  /* } */
  int returned_bytes = getrandom(A, n_of_bytes, 1);
  ++returned_bytes; // dummy operation to avoid not used warning

}


void truncate_array(unsigned char* A, size_t size_A, size_t total_out_bits){
  // A = {B0, B1, ...,B_{size_A -1} }
  // A has n=8*size_A bits
  // We wish to return A with only total_out_bits
  // excessive bits will be stored in memory but their value
  // is 0.

  // we order the bits of A[0], A[1], ..., etc as:
  // b0 b1 ... b7    b8 b9 ... b15   b16 b17 ... b23
  //   A[0]                A[1]           A[3] .....
  // Goal: set all bits bi s.t. i>= total_out_bits to 0
  // e.g. output_size_bits = 4
  // b0 b1 b2 b3 0 ...0

  // all blocks after ceil(output_size_bits/8) should be zero
  // For the singular case when someone choses the output to be zero

  // printf("output_size=%d\n", output_size_bits);
  // see defintion of `rem` below. the extra term is to ensure that ceil adds one
  // the division is exact which mean the last active block has all bits active
  int i =  ceil((float)total_out_bits/8  + ((double) 1 / 16));
  printf("we are going to truncate from i=%d\n", i);
  printf("nbytes=%lu\n", size_A);
  
  for (int j = i; j<size_A; ++j){
    printf("A[%d] was %x \n", j, A[i]);
    A[j] = 0;
    printf("A[%d] is %x \n", j, A[i]);
  }
  // and with output_size_bits ones
  // this convluted formula to deal with the case when all the bits of the last should be active
  // the mod 8 doesn't caputre 8 bits number
  // another solution to add ε < (1/8) inside the ceil within the definiton of i
  int rem = total_out_bits & 7;
  printf("rem=%d bits\n", rem);
  rem = ( 1<<rem) - 1; // 111...1 /
  printf("2^rem - 1=%x\n", rem);
  printf("before A[%d]=%d\n", i-1, A[i-1]);
  A[i - 1] = A[i-1] & rem; // last byte
  printf("after A[%d]=%x\n", i-1, A[i-1]);

}


#pragma omp declare simd uniform( n_of_bits )
void inline truncate_state32bit_get_digest(uint64_t* dst, uint32_t state[8], int n_of_bits){
  /// We extract the digest from ctx and save it in dst
  // uint64_t dst[2] is fixed now // 128 bits, this is a limitation
  // it should be 256 for sha256 :)
  // n_of_bits is how many bits is the compression function output
  

  dst[0] = state[0] + (((uint64_t) state[1])<<32);
  if (n_of_bits < 64){
    uint64_t ones =   (((uint64_t) 1) << (n_of_bits)) - 1;
    dst[0] = dst[0] & ones;
    dst[1] = 0;

    #ifdef VERBOSE_LEVEL
    #if VERBOSE_LEVEL == 3
    printf("ones=%lu\n", ones);
    printf("state[0]=%x, state[1]=%x\n", state[0], state[1]);
    puts("");
    #endif // VERBOSE_LEVEL
    #endif // VERBOSE_LEVEL

    
  } else if (n_of_bits < 128) {
    // since the number of bits is higher or equal 64
    // we need to work on the second element of the dst
    n_of_bits = n_of_bits - 64;

    #ifdef VERBOSE_LEVEL
    uint64_t ones =  ( ((uint64_t) 1<<(n_of_bits)) - 1);
    printf("ones=%lu, n_of_bits=%d\n", ones, n_of_bits);
    printf("state[0]=%x, state[1]=%x\n", state[0], state[1]);
    printf("state[2]=%x, state[2]=%x\n", state[2], state[3]);
    puts("");
    #endif // VERBOSE_LEVEL

    // copy 64bits from state
    dst[1] = state[2] + (((uint64_t) state[3])<<32);
    // truncate it if necessary
    dst[1] = dst[1] & ( ((uint64_t) 1<<(n_of_bits)) - 1);
  } else { // 128 bits limit
    dst[1] = state[2] + (((uint64_t) state[3])<<32);
  }
    
  
}

