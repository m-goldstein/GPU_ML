#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_utils.h"
#include "matrixmul_kernel.cu"
#include "unroll_kernel.cu"

int matrix_equal(matrix_t *A, matrix_t *B, tuple_t ** err) 
{
    if (!A || !A->data || !B || !B->data)
    	return -1;
    if ( !(A->rows == B->rows && A->cols == B->cols))
        return -1;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
	    int offset = i * A->cols + j;
	    if (A->data[offset] != B->data[offset]) {
	        if (!(*err)) {
		    *err = (tuple_t*) malloc(sizeof(tuple_t));
		    (*err)->x = j;
		    (*err)->y = i;
		} else {
		    tuple_t * error = (tuple_t*) malloc(sizeof(tuple_t));
		    error->x = j;
		    error->y = i;
		    error->next = *err;
		    *err = error;
		    
		}
		
	    }
	}
    }
    return 0;
}

int vector_equal(vector_t *A, vector_t *B, tuple4d_t ** err) 
{
    if (!A || !A->data || !B || !B->data)
    	return -1;
    if ( !(A->rows == B->rows && A->cols == B->cols && A->depth == B->depth && A->hyper == B->hyper))
        return -1;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
	    for (int k = 0; k < A->depth; k++) {
	        for (int m = 0; m < A->hyper; m++) {
	             int offset = (m) * (A->rows * A->cols * A->depth) + (k) * (A->rows * A->cols) + (i) * (A->cols) +j;
	             if (A->data[offset] != B->data[offset]) {
	                 if (!(*err)) {
		              *err = (tuple4d_t*) malloc(sizeof(tuple4d_t));
		   	          (*err)->x = j;
		    	      (*err)->y = i;
		    	      (*err)->w = k;
		              (*err)->h = m;
		         } else {
		    	      tuple4d_t * error = (tuple4d_t*) malloc(sizeof(tuple4d_t));
		   	          error->x = j;
		              error->y = i;
		   	          error->w = k;
		              error->h = m;
		              error->next = *err;
		              *err = error;
                    }
		        }
		 }
	    }
       }
   }

    return 0;
}
void forward_pass_cpu(float**** out, float**** in, float**** mask)
{
    //int B = mask->hyper; //batch index
    //int M = in_vec->hyper; //output number of channels
    //int C = in_vec->depth; //input number of channels
    //int H = in_vec->rows; //input rows
    //int W = in_vec->cols; //input cols
    //int K = mask->rows; //mask width (assuming square mask)
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    
    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
	    for (int h = 0; h < H - K + 1; ++h) {
	        for (int w = 0; w < W - K + 1; ++w) {
		   out[b][m][h][w] = 0.0f;

		   for (int c = 0; c < C; ++c) {
		       for (int p = 0; p < K; ++p) {
		           for (int q = 0; q < K; ++q) {
			       out[b][m][h][w] += in[b][c][h+p][w+q] * mask[m][c][p][q];
			   }
		       }
		   }
	       }
	   }
        }
    }
}

void print_tensor(float ****tensor, int m, int c, int h, int w){
    for(int i = 0; i < m; i++) {
	for(int j = 0; j < c; j++) {
	    for (int k = 0; k < h; k++) {
		for (int z = 0; z < w; z++) {
		    printf("%.1f \n", tensor[i][j][k][z]);
		}
	    }
	}
    }
}

void print_vector(vector_t *in)
{
    if (!in || !in->data)
        return;
    for (int i = 0; i < in->rows; i++){
        for (int j = 0; j < in->cols; j++) {
            for (int k = 0; k < in->depth; k++) {
                for (int m = 0; m < in->hyper; m++){
#define v4d(i3, i2, i1, i0) in->data[ (i3) * (in->rows * in->cols * in->depth) + (i2) * (in->rows * in->cols) + (i1) * (in->cols) +(i0)]
                    printf("%.1f \n", v4d(m, k, i, j));
                }
            }
	}
	printf("\n");
    }
    printf("\n");
    #undef v4d
}
float **** alloc_tensor(int batches, int channels, int rows, int cols, matrix_init_t mode)
{
    float **** tensor = (float****) malloc(batches * sizeof(float***));
    for (int i = 0; i < batches; i++) {
	tensor[i] = (float***) malloc(channels * sizeof(float**));
	for (int j = 0; j < channels; j++) {
	    tensor[i][j] = (float**) malloc(rows * sizeof(float*));
	    for (int k = 0; k < rows; k++) {
		tensor[i][j][k] = (float*) malloc(cols *sizeof(float));
	    }
	}
    }
    for (int i = 0; i < batches; i++) {
	for (int j = 0; j < channels; j++) {
	    for (int k = 0; k < rows; k++) {
		for (int m = 0; m < cols; m++) {
		    switch (mode) {
			case IDENTITY:
			    tensor[i][j][k][m] = (k == m) ? 1.0 : 0.0f;
			    break;
			case RAND:
			    tensor[i][j][k][m] = get_rand_val();
			    break;
			case WEIGHTS:
			    tensor[i][j][k][m] = (i % 2) ? 1.0 : 0.5;
			    break;
			case ZEROES:
			default:
			    tensor[i][j][k][m] = 0.0f;
			    break;
			}
		    }
		}
	    }
	}
	return tensor;
}

vector_t* alloc_vector(int batches, int channels, int rows, int cols, matrix_init_t mode)
{
	vector_t* vec = (vector_t*) malloc(sizeof(vector_t));
	vec->rows = rows;
	vec->cols = cols;
	vec->depth = channels;
	vec->hyper = batches;
	vec->size = ( vec->rows ) * (vec->cols) * (vec->depth) * (vec->hyper) * sizeof(float);
	vec->data = (float*) malloc(vec->size);
#define v4d(i3, i2, i1, i0) vec->data[(i3) * (vec->rows * vec->cols * vec->depth) + (i2)*(vec->rows * vec->cols) + (i1) * (vec->cols) +(i0)] 


    
	int i, j, w, h;
    /*int skip;
    int channel_idx;
    float * data;
	for (int b = 0; b < batches; b++) {
        skip = b * vec->rows * vec->cols * vec->depth;
        for(int c = 0; c < channels; c++) {
            channel_idx = c * vec->rows * vec->cols;
            data = vec->data + skip + channel_idx; 
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                    switch (mode) {
                        case IDENTITY:
                            data[j + i * cols] = (i == j) ? 1.0f : 0.0f;
                            break;
                         case RAND:
                            data[j + i * cols] = get_rand_val();
                            break;
                         case WEIGHTS:
                            data[j + i * cols] = (b % 2) ? 1 : 0.5;
                            break;
                         case ZEROES:
                         default:
                            data[j + i * cols] = 0.0f;
                            break;
                    }
                }
            }
        }
    }*/
                
    //int r;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
		    for (w = 0; w < channels; w++) {
		        for (h = 0; h < batches; h++) {
                   switch (mode) {
                        case IDENTITY:
                            v4d(h, w, i, j) = (i == j) ? 1.0f : 0.0f;
                            break;
                        case RAND:
                            //r = get_rand_val();
                            //printf("Assigned: vec[%d][%d][%d][%d] = %d\n", h, w, i, j, r);
                            v4d(h, w, i, j) = get_rand_val();
                            //v4d(h, w, i, j) = r;
                            break;
                        case WEIGHTS:
                            v4d(h, w, i, j) = (w % 2 == 0) ? 0.5 : 1;
                            break;
                        case ZEROES:
                        default:
                            v4d(h, w, i, j) = 0.0f;
                            break;
                    }
                }
            }
        }
	}

	#undef v4d
	return vec;
}
void print_matrix(matrix_t *in) 
{
    if (in->data == NULL || in == NULL)
		return;
	for (int i = 0; i < in->rows; i++) {
		for (int j = 0; j < in->cols; j++) {
			if (j != 0 && j % 25) {
			    printf("%.1f \n", in->data[i * in->cols + j]);
			} else {
				printf("%.1f ", in->data[i * in->cols + j]);
			}
		}
		printf("\n");
	}
}

int get_rand_val()
{
	return rand() % Q + 1;
}

matrix_t* alloc_matrix(int rows, int cols, matrix_init_t mode)
{	
	matrix_t* mat = (matrix_t*) malloc(sizeof(matrix_t));

	mat->rows = rows;
	mat->cols = cols;
	mat->size = rows * cols * sizeof(float);
	mat->data = (float*) malloc(mat->size);

	int i, j, offset;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			offset = j + i * mat->cols;
			switch (mode) {
				case IDENTITY:
					mat->data[offset] = (i == j) ? 1.0f : 0.0f;
					break;
				case RAND:
					mat->data[offset] = get_rand_val();
					break;
				case ZEROES:
				default:
					mat->data[offset] = 0.0f;
					break;
			}
		}
	}

	return mat;
}

void matrix_mult_cpu(float *A, float *B, float *C, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols)
{
    if (A == NULL || B == NULL || C == NULL)
        return;
    int i, j, k;
    for (i = 0; i < a_rows; i++) {
        for (j = 0; j < c_cols; j++) {
            C[i * c_cols + j] = 0.0f;
            for (k = 0; k < b_rows; k++) {
                C[i * c_cols + j] += A[i * a_cols + k] * B[k * b_cols + j];
            }
        }
    }
}

void matrix_mult_cpu(matrix_t * A, matrix_t * B, matrix_t * C)
{
	if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL)
		return;
	int i, j, k;
	for (i = 0; i < A->rows; i++) {
		for (j = 0; j < C->cols; j++) {
			C->data[i* C->cols + j] = 0.0f;
			for (k = 0; k < B->rows; k++) {
				C->data[i * C->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
			}
		}
	}
}
void delete_vector(vector_t * vec)
{
    if (!vec)
        return;
    else {
        free(vec->data);
        free(vec);
    }
}

void delete_matrix(matrix_t * mat)
{
    if (!mat)
        return;
    else {
        free(mat->data);
        free(mat);
    }
}
void delete_tuple(tuple_t * tup)
{
    if (!tup)
        return;
    tuple_t * ref = tup;
    while (tup != NULL) {
        ref = tup->next;
        free(tup);
        tup = ref;
    }
}
void delete_tuple4d(tuple4d_t * tup)
{ 
    if (!tup)
        return;
    tuple4d_t * ref = tup;
    while (tup != NULL) {
        ref = tup->next;
        free(tup);
        tup = ref;
    }
}  
/*
int main(void) 
{
	srand(time(0));
	matrix_t * h_A = alloc_matrix(A_ROWS, A_COLS, RAND);
	matrix_t * h_B = alloc_matrix(B_ROWS, B_COLS, RAND);
	matrix_t * h_C = alloc_matrix(A_ROWS, B_COLS, ZEROES);
	matrix_t * cpu_result = alloc_matrix(A_ROWS, B_COLS, ZEROES);
	float * d_A;
	float * d_B;
	float * d_C;
	cudaMalloc((float**)&d_A, h_A->size);
	cudaMalloc((float**)&d_B, h_B->size);
	cudaMalloc((float**)&d_C, h_C->size);
	cudaMemcpy(d_A, h_A->data, h_A->size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B->data, h_B->size, cudaMemcpyHostToDevice);

        printf("Matrix A: \n");
	print_matrix(h_A);
	printf("\nMatrix B: \n");
	print_matrix(h_B);
	
	printf("Launching kernel...\n");
	matrixMult_launcher(h_A->rows, h_A->cols, h_B->rows, h_B->cols, h_C->rows, h_C->cols, d_A, d_B, d_C);
	cudaMemcpy(h_C->data, d_C, h_C->size, cudaMemcpyDeviceToHost);
	printf("Matrix C: \n");
	print_matrix(h_C);
	printf("\n[CPU] Matrix C: \n");
	matrix_mult_cpu(h_A, h_B, cpu_result);
	print_matrix(cpu_result);
	
	printf("\n\nChecking Results...\n");
	tuple_t * err = (tuple_t*) malloc(sizeof(tuple_t));
	tuple_t * ref = err;
	int status = matrix_equal(h_C, cpu_result, &ref);
	if (status){
	    while (err != NULL) {
	        if (err->x >= 0 && err->x < h_C->cols && err->y >= 0 && err->y < h_C->rows){
	            printf("Error at [%d][%d]\n", err->x, err->y);
		}
		err = err->next;
	    }
	} else {
	    printf("Results are consistent.\n");
	}
	printf("\nGenerating 4D vector representation...\n");
	vector_t * h_V = alloc_vector(A_ROWS, A_COLS, A_DEPTH, A_HYPER, RAND);
	printf("Vector V:\n");
	print_vector(h_V);
	return 0;
}*/
