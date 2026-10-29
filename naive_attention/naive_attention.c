/*
 * This file contains a naive implementation of Scaled Dot-Product Attention. 
 * It may be used as a baseline for faster implementations. 
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdbool.h>
#include<string.h>


#define DEFAULT_N 10
#define DEFAULT_D 5

/*
 * Prints the contents of a 2D array. 
 * Arguments: char* label, int rows, int cols, double arr[rows][cols]
*/
void print_matrix(char* label, int rows, int cols, double arr[rows][cols])
{
    printf("%s:\n", label);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*
 * Performes the calculations for Scaled dot-product attention. 
 * Arguments: int N and int d specify the dimensions of the matrices. 
 * double Q[N][d]: The query matrix
 * double K[N][d]: The key matrix
 * double V[N][d]: The value matrix
 * O[N][d]: Where to write the output to. 
 * Thie function does *not* free the memory for Q, K, V or O. The calling function is responsible for that. 
*/
void naive_attention(int N, int d, double Q[N][d], double K[N][d], double V[N][d], double O[N][d])
{
    // allocate S and P on the heap to allow for larger matrices
    double (*S)[N] = malloc(sizeof(double[N][N]));
    double (*P)[N] = malloc(sizeof(double[N][N]));

    if(!S || !P)
    {
        perror("Error allocating S and P on heap. ");
        free(S);
        free(P);
        exit(1);
    }


    // compute S = Q K^T / sqrt(d)
    double scale = 1.0 / sqrt((double)d);
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            S[i][j] = 0.0;
            for (int k=0; k<d; k++)
            {
                S[i][j] += Q[i][k] * K[j][k];
            }
            S[i][j] *= scale;
        }
    }

    // compute P = softmax(S)
    // Pij = exp(S_ij) / sum(exp(S_ij) for all j)
    for (int i=0; i<N; i++)
    {
        // calculate maximum, for numerical stability. 
        double m = S[i][0];
        for (int j=0; j<N; j++)
        {
            m = S[i][j]>m?S[i][j]:m;
        }

        // calculate exponents and denominator
        double denominator = 0;
        for (int j=0; j<N; j++)
        {
            P[i][j] = exp(S[i][j]-m);
            denominator += P[i][j];
        }

        // normalize
        for (int j=0; j<N; j++)
        {
            P[i][j] /= denominator;
        }

    }

    // compute O = PV
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<d; j++)
        {
            O[i][j] = 0.0;
            for (int k=0; k<N; k++)
            {
                O[i][j] += P[i][k] * V[k][j];
            }
        }
    }

    free(S);
    free(P);
}



int main(int argc, char* argv[])
{
    int N = DEFAULT_N;
    int d = DEFAULT_D;
    bool verbose = true;

    // parse CLI params
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3)
    {
        if (strcmp(argv[3], "--quiet") == 0)
        {
            verbose = false;
        }
        else
        {
            perror("Unexpected argument.");
            exit(2);
        }
    }
    if (argc > 4)
    {
        perror("Too many arguments. ");
        exit(1);
    }

    
    // allocate matrices Q, K, V of shape N x d on the heap
    double (*Q)[d] = malloc(sizeof(double[N][d]));
    double (*K)[d] = malloc(sizeof(double[N][d]));
    double (*V)[d] = malloc(sizeof(double[N][d]));
    double (*O)[d] = malloc(sizeof(double[N][d]));

    // random seed
    srand(42);

    // Init Q, K, V 
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<d; j++)
        {
            Q[i][j] = (double) rand() / RAND_MAX;
            K[i][j] = (double) rand() / RAND_MAX;
            V[i][j] = (double) rand() / RAND_MAX;
        }
    }

    
    if(verbose)
    {
        print_matrix("Q", N, d, Q);
        print_matrix("K", N, d, K);
        print_matrix("V", N, d, V);
    }

    
    naive_attention(N, d, Q, K, V, O);
    

    if(verbose)
    {
        print_matrix("O", N, d, O);
    }

    // free allocated matrices
    free(Q);
    free(K);
    free(V);
    free(O);

    return 0;
}