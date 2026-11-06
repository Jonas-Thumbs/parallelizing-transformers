// flash attention implementation in C

// needed for timer
#define _POSIX_C_SOURCE 200809L

// imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// default values for N, d, Bq, Bk, and quiet
#define DEFAULT_N 10
#define DEFAULT_D 5
#define DEFAULT_BQ 32
#define DEFAULT_BK 64
#define DEFAULT_VERBOSE true

// seed for random number generation
#define SEED 42


// The workspace is allocated once per query tile, and reused for each key/value tile.
  // For parallelization, each thread would have its own.
typedef struct
{
    // Bq x Bk temporary attention scores
    double *scores;
    // Running maximum for each query row
    double *max;
    // Running softmax denominator
    double *sum;
    // Bq x d weighted value sums
    double *accum;   
} Workspace;


// function to return the smaller of two integers
static int smaller(int a, int b)
{
    // Use the ternary operator to return the smaller value
    return a < b ? a : b;
}


// function to compute the elapsed time in milliseconds between two timespec structures
static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000;
}


// function to print a matrix with a given name, 
  // number of rows (N), and number of columns (d)
static void print_matrix(const char *name, int N, int d, double M[N][d])
{
    // print the name of the matrix followed by a colon and a newline
    printf("%s:\n", name);

    // iterate over each row of the matrix
    for (int i = 0; i < N; i++)
    {
        // iterate over each column of the matrix
        for (int j = 0; j < d; j++)
        {
            // print the value of the matrix at the current row and column, 
                // followed by a space
            printf("%f ", M[i][j]);
        }
        // print newline after each row
        printf("\n");
    }
    // print an additional newline after the entire matrix
    printf("\n");
}


// function to reset the workspace for a new query tile
static void reset_workspace(Workspace *work, int rows, int d)
{
    // for each row in the workspace
    for (int i = 0; i < rows; i++)
    {
        // set the running maximum to negative infinity (so that we can find 
            // the maximum in the next step)
        work->max[i] = -INFINITY;
        // set the running softmax denominator to 0 (so that we can compute the 
            // softmax in the next step)
        work->sum[i] = 0.0;

        // set the weighted value sums to 0 for each dimension
        for (int k = 0; k < d; k++)
        {
            work->accum[(size_t)i * d + k] = 0.0;
        }
    }
}


// function to compute the attention scores for a given query tile and key tile
    // Reminder:
        // N is the number of rows in the query and key matrices
        // d is the number of columns in the query and key matrices
        // Q is the query matrix of size N x d
        // K is the key matrix of size N x d
        // q_start is the starting index of the query tile
        // k_start is the starting index of the key tile
        // q_rows is the number of rows in the query tile
        // k_rows is the number of rows in the key tile
        // Bk is the number of columns in the key tile
        // scores is the output matrix of size Bq x Bk
static void compute_scores(int N, int d, double Q[N][d], double K[N][d],
                           int q_start, int k_start, int q_rows, int k_rows,
                           int Bk, double *scores)
{
    // compute the scaling factor for the attention scores
    double scale = 1.0 / sqrt((double)d);

    // iterate over each row in the query tile
    for (int i = 0; i < q_rows; i++)
    {
        // iterate over each row in the key tile
        for (int j = 0; j < k_rows; j++)
        {
            // compute the dot product of the query row and key row to get the attention score
            double score = 0.0;

            for (int k = 0; k < d; k++)
            {
                // accumulate the product of the corresponding elements in the query and key rows
                score += Q[q_start + i][k] * K[k_start + j][k];
            }

            // store the scaled attention score in the output matrix
            scores[(size_t)i * Bk + j] = score * scale;
        }
    }
}

// function to update the softmax values for each query row
static void update_softmax(int N, int d, double V[N][d], int k_start,
                           int q_rows, int k_rows, int Bk, Workspace *work)
{
    // iterate over each row in the query tile
    for (int i = 0; i < q_rows; i++)
    {
        // find the maximum score for the current query row (to prevent overflow
            // when computing the softmax)
        double tile_max = -INFINITY;

        // iterate over each row in the key tile
        for (int j = 0; j < k_rows; j++)
        {
            // get the attention score for the current query row and key row
            double score = work->scores[(size_t)i * Bk + j];

            // update the maximum score if the current score is greater than the 
                // current maximum
            if (score > tile_max)
            {
                tile_max = score;
            }
        }

        // compute the new maximum score for the current query row
            // fmax: returns the maximum of two floating-point numbers
        double new_max = fmax(work->max[i], tile_max);

        // compute the scaling factor for the softmax values based on the new maximum
            // exp: exponential function
        double scale = work->sum[i] == 0.0 ? 0.0 : exp(work->max[i] - new_max);

        // rescale earlier contributions if the running maximum increases
        work->sum[i] *= scale;

        // for each dimension
        for (int k = 0; k < d; k++)
        {
            // rescale the accumulated weighted value sums for the current query row
            work->accum[(size_t)i * d + k] *= scale;
        }

        // for each row in the key tile
        for (int j = 0; j < k_rows; j++)
        {
            // compute the weight for the current query row and key row 
                // based on the attention score and new maximum
            double weight = exp(work->scores[(size_t)i * Bk + j] - new_max);

            // update the running softmax denominator for the current query row
            work->sum[i] += weight;

            // for each dimension, update the accumulated weighted value sums for the current query row
            for (int k = 0; k < d; k++)
            {
                // the part that actually accumulates the weighted values
                work->accum[(size_t)i * d + k] += weight * V[k_start + j][k];
            }
        }

        // update the running maximum for the current query row
        work->max[i] = new_max;
    }
}


// function to write the final output for a query tile
static void write_output(int N, int d, double O[N][d], int q_start,
                         int q_rows, const Workspace *work)
{
    // for each row in the query tile
    for (int i = 0; i < q_rows; i++)
    {
        // for each dimension
        for (int k = 0; k < d; k++)
        {
            // compute the final output value for the current query row and dimension
                // by dividing the accumulated weighted value sum by the running softmax denominator
            O[q_start + i][k] = work->accum[(size_t)i * d + k] / work->sum[i];
        }
    }
}


// ALGO RUN: function that implements the FlashAttention algorithm
void flashattention(int N, int d, double Q[N][d], double K[N][d],
                    double V[N][d], double O[N][d], int Bq, int Bk)
{
    // Allocate workspace for the attention computation
    Workspace work;

    // Allocate memory for the workspace arrays based on the query and key tile sizes
    work.scores = malloc((size_t)Bq * Bk * sizeof(double));
    work.max = malloc((size_t)Bq * sizeof(double));
    work.sum = malloc((size_t)Bq * sizeof(double));
    work.accum = malloc((size_t)Bq * d * sizeof(double));


    // the outer query-tile loop is the future OpenMP parallel-for loop.
    for (int q_start = 0; q_start < N; q_start += Bq)
    {
        // determine the number of rows in the current query tile, 
            // which is the smaller of Bq and the remaining rows in the query matrix
        int q_rows = smaller(Bq, N - q_start);

            // reset the workspace for the current query tile
        reset_workspace(&work, q_rows, d);

        // key/value tiles remain sequential for each query tile
        for (int k_start = 0; k_start < N; k_start += Bk)
        {
            // determine the number of rows in the current key tile, 
                // which is the smaller of Bk and the remaining rows in the key matrix
            int k_rows = smaller(Bk, N - k_start);

            // compute the attention scores for the current query tile and key tile
            compute_scores(N, d, Q, K, q_start, k_start, q_rows, k_rows, Bk, work.scores);
            // update the softmax values for the current query tile based on the 
                // computed attention scores and the corresponding value tile
            update_softmax(N, d, V, k_start, q_rows, k_rows, Bk, &work);
        }

        // just write that to output after
        write_output(N, d, O, q_start, q_rows, &work);
    }

    // Free the allocated workspace memory
    free(work.scores);
    free(work.max);
    free(work.sum);
    free(work.accum);
}


// MAIN FUNCTION: function that serves as the entry point for the program
    // As in, it is the first function that gets called when the program is executed and
        // it is responsible for setting up the program's execution environment, 
        // parsing command-line arguments, and invoking the main logic of the program.

    // structurally, this is laid out the same as the naive attention implementation, 
        // but with some modifications to use the flashattention function instead
        // and incorperate a few more optional parameters that the user can
        // specify on the command line.
int main(int argc, char *argv[])
{
    // Set default values for N, d, Bq, Bk, and quiet based on command-line arguments
        // If no arguments are provided, default values are used
    int N = DEFAULT_N;
    int d = DEFAULT_D;
    int Bq = DEFAULT_BQ;
    int Bk = DEFAULT_BK;
    bool verbose = DEFAULT_VERBOSE;

    // check to make sure the number of arguments does not exceed the expected limit
    if (argc > 8)
    {
        fprintf(stderr, "Error: Too many arguments provided.\n");
        fprintf(stderr, "Usage: %s [N] [d] [--bq Bq] [--bk Bk] [--quiet]\n", argv[0]);
        return 1;
    }

    // Parse command-line arguments to override default values for N, d, Bq, Bk, and quiet
        //IE: set variables based on user input, if provided

        // Parse the first two command-line arguments for N and d
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);

        // Parse additional command-line arguments for Bq, Bk, and quiet flags
            // loop allows for the commands to be in any order
    for (int i = 3; i < argc; i++)
    {
        // check bq
        if (strcmp(argv[i], "--bq") == 0)
        {
            Bq = atoi(argv[++i]);
        }
        // check bk
        else if (strcmp(argv[i], "--bk") == 0)
        {
            Bk = atoi(argv[++i]);
        }
        // check quiet (ie verbosity for printing)
        else if (strcmp(argv[i], "--quiet") == 0)
        {
            verbose = false;
        }
    }
    

    ////////////////////////////////////////////////////////////////////////////
    // Following is a almost direct copy of the naive attention implementation,
        // just with some modifications to use the flashattention function instead
        // and a few lines to calculate the elapsed time for the kernel execution
    /////////////////////////////////////////////////////////////////////////////
    

    // allocate memory for the query, key, value, and output 
        // matrices based on the specified dimensions
    double (*Q)[d] = malloc(sizeof(double[N][d]));
    double (*K)[d] = malloc(sizeof(double[N][d]));
    double (*V)[d] = malloc(sizeof(double[N][d]));
    double (*O)[d] = malloc(sizeof(double[N][d]));

    // generate based on seed
    srand(SEED);

    // fill the query, key, and value matrices with random values between 0 and 1
    for (int i = 0; i < N; i++)
    {
        // for each dimension, generate a random value and assign it to the corresponding 
            // element in the query, key, and value matrices
                // Reminder:
                    // Q is the query matrix of size N x d
                    // K is the key matrix of size N x d
                    // V is the value matrix of size N x d
        for (int j = 0; j < d; j++)
        {
            Q[i][j] = (double)rand() / RAND_MAX;
            K[i][j] = (double)rand() / RAND_MAX;
            V[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // verbose
    if (verbose)
    {
        // then print the matrix
        print_matrix("Q", N, d, Q);
        print_matrix("K", N, d, K);
        print_matrix("V", N, d, V);
    }

    // define timespec to measure the elapsed time for the kernel execution
    struct timespec start, end;

        // start the time
    clock_gettime(CLOCK_MONOTONIC, &start);
        // call the flashattention function to compute the attention output
    flashattention(N, d, Q, K, V, O, Bq, Bk);
        // end the time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // print the results, including the dimensions and the elapsed time 
        // for the kernel execution
    printf("N=%d d=%d Bq=%d Bk=%d kernel_ms=%.3f\n",
           N, d, Bq, Bk, elapsed_ms(start, end));

    // verbose
    if (verbose)
    {
        // print the resulting matrix O after the attention computation
        print_matrix("O", N, d, O);
    }

    // free the allocated memory for the query, key, value, and output matrices
    free(Q);
    free(K);
    free(V);
    free(O);

    // return 0 to indicate successful execution of the program
    return 0;
}
