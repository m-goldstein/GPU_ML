#define A_ROWS 8
#define A_COLS 8
#define A_DEPTH 1
#define A_HYPER 2

#define B_ROWS 4
#define B_COLS 4
#define B_DEPTH 1
#define B_HYPER 2

#define K_ROWS 5
#define K_COLS 5
#define K_DEPTH 1
#define K_HYPER 2

#define TILE_WIDTH 4
#define Q 10

typedef struct {
    int rows;
    int cols;
    float * data;
    size_t size;
} matrix_t;

typedef struct {
    int rows;
    int cols;
    int depth;
    int hyper;
    float * data;
    size_t size;
} vector_t;

typedef struct tuple {
    int x;
    int y;
    struct tuple * next;
} tuple_t;

typedef struct tuple4d {
    int x;
    int y;
    int w;
    int h;
    struct tuple4d * next;
} tuple4d_t;


typedef enum { 
    ZEROES,
    IDENTITY,
    RAND,
    WEIGHTS
} matrix_init_t;    


int get_rand_val();
int matrix_equal(matrix_t * A, matrix_t * B, tuple_t ** err);
int vector_equal(vector_t * A, vector_t * B, tuple4d_t ** err);

void matrix_mult_cpu(float *A, float *B, float *C, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols);
matrix_t * alloc_matrix(int rows, int cols, matrix_init_t mode);

vector_t * alloc_vector(int batches, int channels, int rows, int cols, matrix_init_t mode);

void forward_pass_cpu(vector_t * out_vec, vector_t * in_vec, vector_t * mask);

void print_matrix(matrix_t *in);
void print_vector(vector_t *in);

void delete_vector(vector_t *vec);
void delete_matrix(matrix_t *mat);
void delete_tuple(tuple_t *tup);
void delete_tuple4d(tuple4d_t *tup);
