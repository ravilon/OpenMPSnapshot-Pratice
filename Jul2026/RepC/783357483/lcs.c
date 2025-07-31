#include "lcs.h"

int get_index_of_character(char *str,char x, int len)
{
    for(int i=0;i<len;i++)
    {
        if(str[i]== x)
        {
            return i;
        }
    }
    return -1;//not found the character x in str
}

void print_matrix(int **x, int row, int col)
{
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            printf("%d ",x[i][j]);
        }
        printf("\n");
    }
}



void calc_P_matrix_v2(int **P, char *b, int len_b, char *c, int len_c)
{
    #pragma omp parallel for
    for(int i=0;i<len_c;i++)
    {
        for(int j=1;j<len_b+1;j++)
        {
            if(b[j-1]==c[i])
            {
                P[i][j] = j;
            }
            else
            {
                P[i][j] = P[i][j-1];
            }
        }
    }
}

int lcs_yang_v2(int *DP, int *prev_dp, int **P, char *A, char *B, char *C, int m, int n, int u)
{

    for(int i=1;i<m+1;i++)
    {
        int c_i = get_index_of_character(C,A[i-1],u);
        int t,s;

        #pragma omp parallel for private(t,s) schedule(static)
        for(int j=0;j<n+1;j++)
        {
            t= (0-P[c_i][j])<0;
            s= (0 - (prev_dp[j] - (t*prev_dp[P[c_i][j]-1]) ));

            DP[j] = ((t^1)||(s^0))*(prev_dp[j]) + (!((t^1)||(s^0)))*(prev_dp[P[c_i][j]-1] + 1);
        }

        #pragma omp parallel for schedule(static)
        for(int j=0;j<n+1;j++){
                prev_dp[j] = DP[j];
        }
    }
    return DP[n];
}


int lcs(int **DP, char *A, char *B, int m, int n)
{
//    printf("%s %d \n%s %d\n",A,m,B,n );

    //print_matrix(DP,m+1,n+1);

    for(int i=1;i<(m+1);i++)
    {
        for(int j=1;j<(n+1);j++)
        {
//            if(i==0 || j==0)
//            {
//                DP[i][j]=0;
//            }
            if(A[i-1] == B[j-1])
            {
                DP[i][j] = DP[i-1][j-1] + 1;
            }
            else
            {
                DP[i][j] = max(DP[i-1][j],DP[i][j-1]);
            }
        }
    }

    return DP[m][n];
}