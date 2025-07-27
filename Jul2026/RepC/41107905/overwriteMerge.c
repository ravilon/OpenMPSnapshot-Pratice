/*
* Copyright 2014 Open Connectome Project (http://openconnecto.me)
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
*     http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include<stdint.h>
#include<ndlib.h>
/*#include<omp.h>*/

/*void overwriteMerge ( uint32_t * data1, uint32_t * data2, int dim )*/
/*{*/
    /*int i;*/
/*#pragma omp parallel num_threads(omp_get_max_threads()) */
    /*{*/
/*#pragma omp for private(i) schedule(dynamic)*/
        /*for ( i=0; i<dim; i++)*/
        /*{*/
          /*if ( data2[i] !=0 )*/
            /*data1[i] = data2[i];*/
        /*}*/
    /*}*/
/*}*/

void overwriteMerge ( uint32_t * data1, uint32_t * data2, int dim )
{
    int i;
    for ( i=0; i<dim; i++ )
    {
      if ( data2[i] != 0 )
        data1[i] = data2[i];
    }
}
