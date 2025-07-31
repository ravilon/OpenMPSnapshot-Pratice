/* knn.c - method for a brute-force knn algorithm
 * Copyright (C) 2023  Alexandros Athanasiadis
 *
 * This file is part of knn
 *
 * knn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * knn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "knn.h"

#include <omp.h>

#include "sedm.h"
#include "qselect.h"
#include "matrix.h"
#include "def.h"

/* knn computes the K nearest neighbours in Y of each element in X
 *
 * inputs:
 * X (elem_t *): matrix stored in row-major format that was created using create_matrix
 * n (size_t): rows of matrix X
 * Y (elem_t *): matrix stored in row-major format that was created using create_matrix
 *
 * Y_begin (size_T): real index of the first element in Y.
 *                   if the entire Y is given at once this should be 0
 *
 * m (size_t): rows of matrix Y
 * d (size_t): columns of matrices X,Y
 * k (size_t): the number of smallest elements to find for each X row
 * t (size_t): the maximum number of rows in X to be processed at a time
 *
 * prev_result (knn_result **): pointer to the previous knn result.
 *                              if no previous result exists this should
 *                              point to NULL.
 *
 * returns:
 * knn_result struct of size n * k.
 */
knn_result *knn(
	const elem_t *X, size_t n, const elem_t *Y, size_t Y_begin, size_t m, 
	size_t d, size_t k, size_t t, knn_result **prev_result) {

	/* the result will be stored in res */
	knn_result *res = (knn_result *) malloc(sizeof(knn_result));

	res->m = n;
	res->k = k;

	res->n_idx = (size_t *) malloc(n * k * sizeof(size_t));
	res->n_dist = (elem_t *) malloc(n * k * sizeof(elem_t));

	/* setting up OpenMP to use the maximum available threads */
	int threadnum = omp_get_max_threads();
	omp_set_num_threads(threadnum);

	/* if t is larger than n then set t = n */
	t = min(t, n);

	/* arrays that will store the computed SEDM and indices */
	elem_t *D = (elem_t *) malloc(t * m * sizeof(elem_t));
	size_t *ind = (size_t *) malloc(t * m * sizeof(size_t));

	/* slice X into parts of size t in order to not use a lot of memory. */
	for(size_t X_begin = 0 ; X_begin < n ; X_begin += t) {
		size_t X_end = min(X_begin + t, n);

		size_t slice_size = X_end - X_begin;

		const elem_t *X_slice = MATRIX_ROW(X, X_begin, n, d);

		/* calculate the SEDM for X_slice and Y */
		sedm(X_slice, slice_size, Y, m, d, D);

		/* in parallel for all rows in the slice perform knn */
		#pragma omp parallel for
		for(int tid = 0 ; tid < slice_size ; tid++) {
			elem_t *Di = MATRIX_ROW(D, tid, t, m);
			size_t *ind_i = MATRIX_ROW(ind, tid, t, m);

			gen_indices(ind_i, Y_begin, m);

			/* using QuickSelect to find the k smallest elements in Di */
			qselect(k, Di, ind_i, m);

			/* sortthe first k elements in Di using BubbleSort */
			for(size_t i = 0 ; i < k - 1 ; i++) {
				for(size_t j = 0 ; j < k - i - 1 ; j++) {
					if(Di[j] > Di[j + 1]) {
						SWAP(Di[j], Di[j + 1]);
						SWAP(ind_i[j], ind_i[j + 1]);
					}
				}
			}


			if(*prev_result == NULL) {
			/* if there is no previous result, store the results in res */
				for(size_t j = 0 ; j < k ; j++) {
					MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = Di[j];
					MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = ind_i[j];
				}
			} else {
			/* if there is a previous result, merge the 2 results and store in res */
				elem_t *prev_dist = MATRIX_ROW((*prev_result)->n_dist, X_begin + tid, n, k);
				size_t *prev_idx = MATRIX_ROW((*prev_result)->n_idx, X_begin + tid, n, k);

				/* initialize pointers to the beggining of both arrays */
				size_t p_idx = 0;
				size_t d_idx = 0;

				/* we need the k smallest elements of both arrays */
				for(size_t j = 0 ; j < k ; j++) {
					/* select the smallest of the pointer elements, 
					 * then increment that pointer */
					if(prev_dist[p_idx] <= Di[d_idx]) {
						MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = prev_dist[p_idx];
						MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = prev_idx[p_idx];

						p_idx += 1;

					} else {
						MATRIX_ELEM(res->n_dist, X_begin + tid, j, n, k) = Di[d_idx];
						MATRIX_ELEM(res->n_idx, X_begin + tid, j, n, k) = ind_i[d_idx];

						d_idx += 1;
					}
				}
			}
		}
	}

	/* free the temporary arrays */
	free(D);
	free(ind);

	/* delete the previous result */
	if(*prev_result != NULL)
		delete_knn(*prev_result);

	return res;
}

/* frees the memory for a knn_result struct */
void delete_knn(knn_result *knn) {
	free(knn->n_idx);
	free(knn->n_dist);

	free(knn);
}
