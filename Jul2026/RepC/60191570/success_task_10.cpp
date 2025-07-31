/*--------------------------------------------------------------------
  (C) Copyright 2006-2013 Barcelona Supercomputing Center
                          Centro Nacional de Supercomputacion
  
  This file is part of Mercurium C/C++ source-to-source compiler.
  
  See AUTHORS file in the top level directory for information
  regarding developers and contributors.
  
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  Mercurium C/C++ source-to-source compiler is distributed in the hope
  that it will be useful, but WITHOUT ANY WARRANTY; without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the GNU Lesser General Public License for more
  details.
  
  You should have received a copy of the GNU Lesser General Public
  License along with Mercurium C/C++ source-to-source compiler; if
  not, write to the Free Software Foundation, Inc., 675 Mass Ave,
  Cambridge, MA 02139, USA.
--------------------------------------------------------------------*/



/*
<testinfo>
test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
</testinfo>
*/


#include <stdio.h>
#include <stdlib.h>

#pragma omp target device(smp)
#pragma omp task out(a[n])
void generator(int *a, int n)
{
    fprintf(stderr, "%s: a -> %p | a[%d] -> %p\n", __FUNCTION__, a, n, &a[n]);
    a[n] = n;
}

#pragma omp target device(smp)
#pragma omp task in(*a)
void consumer(int *a, int n)
{
    fprintf(stderr, "%s a[%d] -> %p\n", __FUNCTION__, n, &a[n]);
    if (*a != n)
    {
        fprintf(stderr, "%d != %d\n", *a, n);
        abort();
    }
}

#define SIZE 10
int k[SIZE] = { 0 };

int main(int argc, char* argv[])
{

    int i;

    int *p;
    for (i = 0; i < SIZE; i++)
    {
        generator(k, i); // k[i] = i
        p = &k[i];
        consumer(p, i);
    }

#pragma omp taskwait
}
