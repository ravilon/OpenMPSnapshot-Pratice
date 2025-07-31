// C++ implementation of the above approach
#include <iostream>
using namespace std;

/* m*n is the board dimension
k is the number of knights to be placed on board
solutions is the number of possible solutions */
int m, n, k;
int solutions = 0;

/* This function is used to create an empty m*n board */
void makeBoard(char** board)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			board[i][j] = '_';
		}
	}
}

void displayBoard(char** board)
{
	char* charArray = new char[(((n * 3) + 1) * m) + 1];
	std::fill(charArray, charArray + (((n * 3) + 1) * m) + 1, ' ');
	//#pragma omp parallel for shared(charArray)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			charArray[((i * n * 3)) + (3 * j + 1)] = board[i][j];
		}
		charArray[(i * n * 3) + ((3 * n) - 1)] = '\n';
	}
	charArray[(((n * 3) + 1) * m)] = '\n';
	cout << charArray;
}


/* This function marks all the attacking
position of a knight placed at board[i][j]
position */
void attack(int i, int j, char a, char** board)
{

	/* conditions to ensure that the
	block to be checked is inside the board */
	if ((i + 2) < m && (j - 1) >= 0) {
		board[i + 2][j - 1] = a;
	}
	if ((i - 2) >= 0 && (j - 1) >= 0) {
		board[i - 2][j - 1] = a;
	}
	if ((i + 2) < m && (j + 1) < n) {
		board[i + 2][j + 1] = a;
	}
	if ((i - 2) >= 0 && (j + 1) < n) {
		board[i - 2][j + 1] = a;
	}
	if ((i + 1) < m && (j + 2) < n) {
		board[i + 1][j + 2] = a;
	}
	if ((i - 1) >= 0 && (j + 2) < n) {
		board[i - 1][j + 2] = a;
	}
	if ((i + 1) < m && (j - 2) >= 0) {
		board[i + 1][j - 2] = a;
	}
	if ((i - 1) >= 0 && (j - 2) >= 0) {
		board[i - 1][j - 2] = a;
	}
}

/* If the position is empty,
place the knight */
bool canPlace(int i, int j, char** board)
{
	if (board[i][j] == '_')
		return true;
	else
		return false;
}

/* Place the knight at [i][j] position
on board */
void place(int i, int j, char k, char a, char** board,
	char** new_board)
{

	/* Copy the configurations of
	old board to new board */
	for (int y = 0; y < m; y++) {
		for (int z = 0; z < n; z++) {
			new_board[y][z] = board[y][z];
		}
	}

	/* Place the knight at [i][j]
	position on new board */
	new_board[i][j] = k;

	/* Mark all the attacking positions
	of newly placed knight on the new board */
	attack(i, j, a, new_board);
}

/* Function for placing knights on board
such that they don't attack each other */
void kkn(int k, int sti, int stj, char** board)
{

	/* If there are no knights left to be placed,
	display the board and increment the solutions */
		if (k == 0) {
			displayBoard(board);
			#pragma omp critical
			solutions++;
		}
		else {

			/* Loop for checking all the
	positions on m*n board */
			
				for (int i = sti; i < m; i++) {
					for (int j = stj; j < n; j++) {

						/* Is it possible to place knight at
		[i][j] position on board? */
						if (canPlace(i, j, board)) {

							#pragma omp task firstprivate(i, j, k)
							{
								/* Create a new board and place the
								new knight on it */
								char** new_board = new char* [m];
								for (int x = 0; x < m; x++) {
									new_board[x] = new char[n];
								}
								place(i, j, 'K', 'A', board, new_board);

								/* Call the function recursively for
								(k-1) leftover knights */
								kkn(k - 1, i, j, new_board);

								/* Delete the new board
								to free up the memory */
								for (int x = 0; x < m; x++) {
									delete[] new_board[x];
								}
								delete[] new_board;
							}
						}
					}
					stj = 0;
				}
				#pragma omp taskwait

		}
	
}

// Driver code
int main()
{
	m = 6, n = 6, k = 3;

	/* Creation of a m*n board */
	char** board = new char* [m];
	for (int i = 0; i < m; i++) {
		board[i] = new char[n];
	}

	/* Make all the places are empty */
	makeBoard(board);
	#pragma omp parallel
	{
		#pragma omp single
		{
			kkn(k, 0, 0, board);
		}
	}

	for (int i = 0; i < m; i++) {
		delete[] board[i];
	}
	delete[] board;
	cout << endl << "Total number of solutions : " << solutions;
	return 0;
}

// Reference: https://www.geeksforgeeks.org/place-k-knights-such-that-they-do-not-attack-each-other/