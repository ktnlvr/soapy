#pragma once

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdio.h>

void matrix_inverse(gsl_matrix *A, gsl_matrix *Ainv) {
  int n = A->size1;

  gsl_matrix *tmp = gsl_matrix_alloc(n, n);
  gsl_matrix_memcpy(tmp, A);

  gsl_permutation *p = gsl_permutation_alloc(n);
  int sign;

  gsl_linalg_LU_decomp(tmp, p, &sign);
  gsl_linalg_LU_invert(tmp, p, Ainv);

  gsl_permutation_free(p);
  gsl_matrix_free(tmp);
}

void matrix_sqrt(gsl_matrix *A, gsl_matrix *result) {
  int n = A->size1;

  gsl_vector *eval = gsl_vector_alloc(n);
  gsl_matrix *evec = gsl_matrix_alloc(n, n);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);

  gsl_matrix *tmp = gsl_matrix_alloc(n, n);
  gsl_matrix_memcpy(tmp, A);

  gsl_eigen_symmv(tmp, eval, evec, w);
  gsl_eigen_symmv_free(w);

  gsl_matrix *D = gsl_matrix_calloc(n, n);

  for (int i = 0; i < n; i++)
    gsl_matrix_set(D, i, i, sqrt(gsl_vector_get(eval, i)));

  gsl_matrix *temp = gsl_matrix_alloc(n, n);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec, D, 0.0, temp);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp, evec, 0.0, result);

  gsl_matrix_free(tmp);
  gsl_matrix_free(D);
  gsl_matrix_free(temp);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}
