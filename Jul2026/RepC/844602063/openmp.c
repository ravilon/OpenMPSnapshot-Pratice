#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define MAX_SIZE 100

typedef struct {
    int rows;
    int columns;
    double data[MAX_SIZE][MAX_SIZE];
} Matrix;

Matrix matrix_addition(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix result;
    result.rows = matrix1->rows;
    result.columns = matrix1->columns;

    #pragma omp parallel for
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            result.data[i][j] = matrix1->data[i][j] + matrix2->data[i][j];
        }
    }

    return result;
}

Matrix matrix_subtraction(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix result;
    result.rows = matrix1->rows;
    result.columns = matrix1->columns;

    #pragma omp parallel for
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            result.data[i][j] = matrix1->data[i][j] - matrix2->data[i][j];
        }
    }

    return result;
}

Matrix matrix_multiplication(const Matrix* matrix1, const Matrix* matrix2) {
    Matrix result;
    result.rows = matrix1->rows;
    result.columns = matrix2->columns;

    #pragma omp parallel for
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix2->columns; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < matrix1->columns; k++) {
                result.data[i][j] += matrix1->data[i][k] * matrix2->data[k][j];
            }
        }
    }

    return result;
}

Matrix matrix_inversion(const Matrix* matrix) {
    int n = matrix->rows;

    Matrix augmented;
    augmented.rows = n;
    augmented.columns = 2 * n;

    // Create an augmented matrix [A | I]
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented.data[i][j] = matrix->data[i][j];
        }
        for (int j = n; j < 2 * n; j++) {
            augmented.data[i][j] = (j == (i + n)) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double pivot = augmented.data[i][i];
        // Divide the current row by the pivot element
        for (int j = i; j < 2 * n; j++) {
            augmented.data[i][j] /= pivot;
        }
        // Eliminate other elements in the column
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = augmented.data[j][i];
                for (int k = i; k < 2 * n; k++) {
                    augmented.data[j][k] -= factor * augmented.data[i][k];
                }
            }
        }
    }

    // Extract the inverted matrix from the augmented matrix
    Matrix inverted;
    inverted.rows = n;
    inverted.columns = n;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverted.data[i][j] = augmented.data[i][j + n];
        }
    }

    return inverted;
}

void print_matrix(const Matrix* matrix) {
	printf("\n");
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            printf(" %.2f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int size[4];
    printf("\nENTER NUMBER OF ROWS FOR MATRIX A: ");
    scanf("%d",&size[0]);
    printf("ENTER NUMBER OF COLUMNS FOR MATRIX A: ");
    scanf("%d",&size[1]);
    printf("ENTER NUMBER OF ROW FOR MATRIX B:");
    scanf("%d",&size[2]);
    printf("ENTER NUMBER OF COLUMNS FOR MATRIX B: ");
    scanf("%d",&size[3]);
    Matrix matrix1, matrix2;
    
    matrix1.rows = size[0];
    matrix1.columns = size[1];
    matrix2.rows = size[2];
    matrix2.columns = size[3];
    
	double value;
	printf("\nENTER VALUES FOR MATRIX A\n");
	
	for(int r=0;r<size[0];r++)
	{
		for(int c=0;c<size[1];c++)
		{
			printf("in ROW %d and COLUMN %d: ",r,c);
			scanf("%lf",&value);
			matrix1.data[r][c] = value;
		}
	}
	printf("\nENTER VALUES FOR MATRIX B\n");
	
	for(int r=0;r<size[2];r++)
	{
		for(int c=0;c<size[3];c++)
		{
			printf("in ROW %d and COLUMN %d: ",r,c);
			scanf("%lf",&value);
			matrix2.data[r][c] = value;
		}
	}
	printf("\nMatrix A:\n");
    print_matrix(&matrix1);

    printf("\nMatrix B:\n");
    print_matrix(&matrix2);
    int menu;
    while(true){
		printf("\nOPERATIONS MENU - SELECT AN OPTION");
		printf("1.Addition\n2.Subtraction\n3.Multiplication\n4.Inversion\n5.EXIT");
		scanf("%d",menu);
		
	// Matrix addition & subtraction
    if(matrix1.rows == matrix2.rows && matrix1.columns == matrix2.columns && menu == 1)
    {
		
		Matrix result_addition = matrix_addition(&matrix1, &matrix2);
		printf("\nMATRIX ADDITION:\n");
		print_matrix(&result_addition);

		Matrix result_subtraction = matrix_subtraction(&matrix1, &matrix2);
		printf("\nMATRIX SUBSTRACTION:\n");
		print_matrix(&result_subtraction);
	}
	else if(matrix1.rows == matrix2.rows && matrix1.columns == matrix2.columns)
    {
		Matrix result_subtraction = matrix_subtraction(&matrix1, &matrix2);
		printf("\nMATRIX SUBSTRACTION:\n");
		print_matrix(&result_subtraction);
	}
	else if (menu == 1) 
	{
		printf("\n\t\t***************** Matrices can't be added *********************\n");
    }
    else if (menu == 2)
    {
		printf("\n\t\t***************** Matrices can't be subtracted *********************\n");
    }
    // Matrix multiplication
    if(matrix1.columns == matrix2.rows & menu == 3)
    {
		Matrix result_multiplication = matrix_multiplication(&matrix1, &matrix2);
		printf("\nMATRIX MULTIPLICATION:\n");
		print_matrix(&result_multiplication);
	}
	else if(menu == 3)
	{
		printf("\n\t\t********************** Matrices can't be multiplied *************************\n");
	}
    
    // Matrix inversion
	Matrix Matx1Inversion = matrix_inversion(&matrix1);
	Matrix Matx2Inversion = matrix_inversion(&matrix2);
	int invA, invB; 
    
	for(int r=0;r<size[0];r++)
	{
		for(int c=0;c<size[1];c++)
		{
			if(Matx1Inversion.data[r][c] == INFINITY || Matx1Inversion.data[r][c] == -INFINITY || 	 
			isnan(Matx1Inversion.data[r][c]))
    		{
    			invA=1;
   			}
   			else if(invA !=1)
   			{
				invA=0;      			
    		}
		}
	}
	
	for(int r=0;r<size[2];r++)
	{
		for(int c=0;c<size[3];c++)
		{
			if(Matx1Inversion.data[r][c] == INFINITY || Matx1Inversion.data[r][c] == -INFINITY || 	 
			isnan(Matx1Inversion.data[r][c]))
    		{
    			invB=1;
   			}
   			else if(invB!=1)
   			{
				invB=0;
    		}
		}
	}
	if(invA == 1 && menu == 4){
		printf("\n\t\t*********************** Matrix A is Non-Invertable ***************************\n");
	}
	else if(invA == 0 && menu == 4){
		printf("\nINVERTED MATRIX A\n");
        print_matrix(&Matx1Inversion);
	}
	if(invB== 1 && menu == 4){
		printf("\n\t\t*********************** Matrix B is Non-Invertable ***************************\n");
	}
	else if(invB == 0 && menu == 4){
		printf("\nINVERTED MATRIX B:\n");
        print_matrix(&Matx2Inversion);
	}
	if(menu == 5){
		break;
	}
	}
}
