#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define FPS 24;
#define CELLSIZE 5;
#define PROBABILITY 10;
const int screenWidth = 800;
const int screenHeight = 450;
int cols, rows;
int cellSize;
Vector2 startPos = {0, 0};

void print_usage()
{
	printf("Usage: myprogram -c <cell_size> -p <probability> -f <fps>\n");
    printf("  -c <cell_size>    : Cell size (1-10)\n");
    printf("  -p <probability>  : Probability (1-100)\n");
    printf("  -f <fps>          : Frames per second (1-300)\n");
}

int mod(int x, int y)
{
	return (x % y + y) % y;
}
int neighbours(int i, int j, int **board)
{
	int n = 0;
	for (int l = i - 1; l <= i + 1; l++)
	{
		for (int m = j - 1; m <= j + 1; m++)
		{
			if (!((i == l) && (j == m)) && board[mod(l, rows)][mod(m, cols)])
			{
				n++;
			}
		}
	}
	return n;
}

void render(int **board)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (board[i][j])
				DrawRectangleV((Vector2){startPos.x + j * cellSize,
										 startPos.y + i * cellSize},
							   (Vector2){cellSize, cellSize},
							   RAYWHITE);
		}
	}
}
void compute_next_state(int start, int step, int **board, int **next_board)
{
	for (int i = start; i < rows; i = i + step)
	{
		for (int j = 0; j < cols; j++)
		{
			int n = neighbours(i, j, board);
			if (board[i][j])
			{
				next_board[i][j] = (n == 2 || n == 3);
			}
			else
			{
				next_board[i][j] = (n == 3);
			}
		}
	}
}

void push_next_state(int start, int step, int **board, int **next_board)
{
	for (int i = start; i < rows; i = i + step)
	{
		for (int j = 0; j < cols; j++)
		{
			board[i][j] = next_board[i][j];
		}
	}
}

void randomInitState(__uint8_t percentage, int **board)
{
	srand(time(NULL));
	if (percentage > 100)
		percentage = 100;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (rand() % 100 < percentage)
			{
				board[i][j] = true;
			}
		}
	}
}
int main(int argc, char *argv[])
{
	int opt;
	int cellSizeArg, probabilityArg, fpsArg;
	cellSizeArg = 0;
	probabilityArg = 0;
	fpsArg = 0;

	while ((opt = getopt(argc, argv, "c:p:f:")) != -1)
	{
		switch (opt)
		{
		case 'c':
			cellSizeArg = atoi(optarg);
			break;
		case 'p':
			probabilityArg = atoi(optarg);
			break;
		case 'f':
			fpsArg = atoi(optarg);
			break;
		default:
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	cellSize = (cellSizeArg > 0 && cellSizeArg <= 10) ? cellSizeArg : CELLSIZE;
	int probablility = (probabilityArg > 0 && probabilityArg <= 100) ? probabilityArg : PROBABILITY;
	int fps = (fpsArg > 0 && fpsArg <= 300) ? fpsArg : FPS;

	cols = screenWidth / cellSize;
	rows = screenHeight / cellSize;
	startPos.x = screenWidth * .5 - (cols * cellSize) * 0.5;
	startPos.y = screenHeight * .5 - (rows * cellSize) * 0.5;

	SetConfigFlags(FLAG_VSYNC_HINT);

	int **board = malloc(rows * sizeof(int *));
	int **next_board = malloc(rows * sizeof(int *));
	for (int i = 0; i < rows; i++)
	{
		board[i] = (int *)calloc(cols, sizeof(int));
		next_board[i] = (int *)calloc(cols, sizeof(int));
	}

	randomInitState(probablility, board);

	// glider(board);

	InitWindow(screenWidth, screenHeight, "Game Of Life");
	SetTargetFPS(fps);

	while (!WindowShouldClose())
	{

		BeginDrawing();
		ClearBackground(BLACK);
		render(board);
		#pragma omp parallel
		{
			int id = omp_get_thread_num();
			int thrds = omp_get_num_threads();
			compute_next_state(id, thrds, board, next_board);
			#pragma omp barrier
			push_next_state(id, thrds, board, next_board);
		}
		EndDrawing();
	}

	for (int i = 0; i < rows; i++)
	{
		free(board[i]);
		free(next_board[i]);
	}
	CloseWindow();
}

void glider(int **board)
{
	board[0][1] = 1;
	board[1][2] = 1;
	board[2][0] = 1;
	board[2][1] = 1;
	board[2][2] = 1;
}
