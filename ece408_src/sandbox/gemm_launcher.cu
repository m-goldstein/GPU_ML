#include "cpu_utils.cu"
int main() 
{
    printf("init host vectors\n");
    //vector_t * x = alloc_vector (A_HYPER, A_DEPTH, A_ROWS, A_COLS, RAND);
    vector_t * x = (vector_t *) malloc(sizeof(vector_t));
    vector_t * w = alloc_vector (K_HYPER, A_DEPTH, K_ROWS, K_COLS, IDENTITY);
    vector_t * y = alloc_vector (A_DEPTH, A_DEPTH, A_ROWS - K_ROWS + 1, A_COLS - K_COLS + 1, WEIGHTS);
    matrix_t * x_unrolled = alloc_matrix(A_DEPTH * K_ROWS * K_COLS, (A_ROWS - K_ROWS + 1) * (A_COLS - K_COLS + 1), ZEROES);
    float * h_x = x->data;
    float * h_w = w->data;
    float * h_y = y->data;
    float * h_x_unrolled = x_unrolled->data;
    float * d_x;
    float * d_x_unrolled;
    float * d_w;
    float * d_y;
    cudaMalloc((float**)&d_x, x->size);
    cudaMalloc((float**)&d_w, w->size);
    cudaMalloc((float**)&d_y, y->size);
    cudaMemcpy(d_x, x->data, x->size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w->data, w->size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y->data, y->size, cudaMemcpyHostToDevice);
    int C = A_DEPTH;
    int B = A_HYPER;
    int H = A_ROWS;
    int W = A_COLS;
    int K = K_COLS;
    cudaMalloc((float**)&d_x_unrolled, sizeof(float) * C * K * K * (W - K + 1) * (H - K + 1));
    printf("init device vectors\nunrolling..");
    unroll_input(C, H, W, K, 0, d_x, d_x_unrolled);
    cudaMemcpy(h_x_unrolled, d_x_unrolled, x_unrolled->size, cudaMemcpyDeviceToHost);
    printf("\ndone\n");
    printf("Vector: \n");
    print_vector(x);
    printf("Unrolled matrix: \n");
    print_matrix(x_unrolled);
    printf("init matrix multiplication\n");
    for(int b = 0; b < B; b++) {
        matrixMult(K_HYPER, C, K, b, H - K + 1, W - K + 1, d_w, d_x_unrolled, d_y);
    }

    cudaMemcpy(h_y, d_y, y->size, cudaMemcpyDeviceToHost);
    printf("done");
    printf("Input matrix: \n");
    print_vector(x);
    printf("Weight matrix: \n");
    print_vector(w);
    printf("\nOutput Matrix: \n");
    print_vector(y);
    vector_t * cpu_out = alloc_vector (K_HYPER, A_DEPTH, A_ROWS - K_ROWS + 1, A_COLS - K_COLS + 1, WEIGHTS);
    matrix_mult_cpu(x_unrolled->data, w->data, cpu_out->data, x_unrolled->rows, x_unrolled->cols, w->rows, w->cols, x_unrolled->rows, w->cols);
    printf("[CPU] Output Matrix: \n");
    print_vector(cpu_out);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_x_unrolled);
    delete_vector(x);
    delete_vector(y);
    delete_vector(w);
    delete_matrix(x_unrolled);
    return 0;
}
