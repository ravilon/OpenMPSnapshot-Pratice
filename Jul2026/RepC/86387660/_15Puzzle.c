/*
	Name: Bilal (K13-2314)
	Program: 15 Puzzle Heuristic Approach using OpenMP.
	Compile: g++ _15Puzzle.c -std=c++11 -fopenmp -o _15Puzzle
														 		*/

// Libraries
#include <iostream>
	using std::cout;
	using std::endl;
#include <algorithm>
	using std::binary_function;
	using std::equal;
	using std::copy;
#include <queue>  
	using std::priority_queue;
	using std::deque;   
#include <set>
	using std::set;
#include <iomanip>
	using std::setw;
	using std::setfill;
#include <ctime>
	// For Parallel Programming (OpenMP)
#include <omp.h>

// Max Node Limit
#define MAX_NODE 10000

struct Node
{
	int state[4][4];
	Node *ancestor;
	int cost, heuristic;

	void calHeuristic();
	void PrintState();
};
void Node::calHeuristic()
{
	int row[] = { 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3 };
	int col[] = { 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2 };

	int r = 0;
	for (int i = 0; i<4; i++)
		for (int j = 0; j<4; j++)
			if (state[i][j] != 0)
				r += abs(row[state[i][j]] - i) + abs(col[state[i][j]] - j);
	heuristic = r;
}
void Node::PrintState()
{
	for (int i = 0; i < 4; i++) {
		cout << endl
			<< "    " << setw(2) << setfill('0') << state[i][0] 
			<< "  |  " << setw(2) << setfill('0') << state[i][1] 
			<< "  |  " << setw(2) << setfill('0') << state[i][2] 
			<< "  |  " << setw(2) << setfill('0') << state[i][3] 
			<< "  " << endl;
		if (i != 3)
			cout << "  ------ ------ ------ ------ ";
	}
	cout << endl;
}

struct PQSorter
{
	bool operator() (Node *lhs, Node * rhs)
	{
		return lhs->cost + lhs->heuristic > rhs->cost + rhs->heuristic;
	}
};
priority_queue<Node*, deque<Node*>, PQSorter> Queue;
set <Node*> Nodes;

struct CompareVPtrs : public binary_function<Node*, Node*, bool>
{
	bool operator()(Node *lhs, Node *rhs) const
	{
		return equal((int *)lhs->state, (int *)lhs->state + 16,
			(int *)rhs->state);
	}
}
CompareVP;

void LocateSpace(int& irRow, int& irCol, int state[4][4]) {
	for (int iRow = 0; iRow < 4; ++iRow) {
		for (int iCol = 0; iCol < 4; ++iCol) {
			if (state[iRow][iCol] == 0) {
				irRow = iRow;
				irCol = iCol;
			}
		}
	}
}
void Move(int state[4][4], int move) {
	int iRowSpace;
	int iColSpace;
	LocateSpace(iRowSpace, iColSpace, state);
	int iRowMove(iRowSpace);
	int iColMove(iColSpace);
	switch (move) {
		case 0:
			iRowMove = iRowSpace + 1;
			break;
		case 1:
			iRowMove = iRowSpace - 1;
			break;
		case 2:
			iColMove = iColSpace + 1;
			break;
		case 3:
			iColMove = iColSpace - 1;
			break;
	}
	// Make sure that the square to be moved is in bounds
	if (iRowMove >= 0 && iRowMove < 4 && iColMove >= 0 && iColMove < 4) {
		state[iRowSpace][iColSpace]	= state[iRowMove][iColMove];
		state[iRowMove][iColMove]	= 0;
	} 
}
void Randomize(int state[4][4]) {
	srand((unsigned int)time(NULL));
	for (int iIndex = 0; iIndex < 80; ++iIndex) {			// 0 to 80 Times Random Move
		int randomMove = (rand() % 4);
		switch (randomMove) {
			case 0:
				Move(state, randomMove);
				break;
			case 1:
				Move(state, randomMove);
				break;
			case 2:
				Move(state, randomMove);
				break;
			case 3:
				Move(state, randomMove);
				break;
		}
	}
}

// Initialize Puzzle -Function
void Initialize()
{
	Node* startingState = new Node;
	int inputPuzzle[4][4] = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 0 }
	};
	// Randomize the Input Puzzle
	Randomize(inputPuzzle);
	cout << "15 Puzzle Input (Random): " << endl;
	for (int i = 0; i < 4; i++) {
		cout << endl
			<< "    " << setw(2) << setfill('0') << inputPuzzle[i][0] 
			<< "  |  " << setw(2) << setfill('0') << inputPuzzle[i][1] 
			<< "  |  " << setw(2) << setfill('0') << inputPuzzle[i][2] 
			<< "  |  " << setw(2) << setfill('0') << inputPuzzle[i][3] 
			<< "  " << endl;
		if (i != 3)
			cout << "  ------ ------ ------ ------ ";
	}
	// Parallel Loop
	#pragma omp parallel num_threads(4)
	{
		int iRow = omp_get_thread_num();
		for (int iCol = 0; iCol < 4; ++iCol) {
			startingState->state[iRow][iCol] = inputPuzzle[iRow][iCol];
		}
	}
	// Start Puzzle Here
	startingState->cost = 0;
	startingState->calHeuristic();
	startingState->ancestor = NULL;		// First Node, -NULL
	Queue.push(startingState);
	Nodes.insert(startingState);
	cout << endl;
}

/*
	The AddNeighbour Function Solve The 15 Puzzle If Posssible...
																	*/

void AddNeighbour(Node* gameState)
{
	int rowSpace, colSpace;
	#pragma omp for
	for (int i = 0;i<4;i++)
		for (int j = 0;j<4;j++)
			if (gameState->state[i][j] == 0)
			{
				rowSpace = i; colSpace = j; break;
			}

	int di[] = { -1, 0, 1, 0 }, dj[] = { 0, -1, 0, 1 };

	// Parallel for Move (Up, Down, Left, Right) -- 4 Threads
	#pragma omp parallel num_threads(4)
	{
		int k = omp_get_thread_num();		// Threads 0, 1, 2, 3 Like For Loop
		int i = rowSpace + di[k];
		int j = colSpace + dj[k];
		if (i >= 0 && j >= 0 && i <= 3 && j <= 3)
		{
			Node* n = new Node;
			copy((int*)gameState->state, (int*)gameState + 16, (int*)n->state);
			n->state[i][j] = 0;
			n->state[rowSpace][colSpace] = gameState->state[i][j];
			n->cost = gameState->cost + 1;
			n->calHeuristic();
			n->ancestor = gameState;
			// Critical Region (Lock1)
			#pragma omp critical (lock1)		// Critical Region
			{
				if (find_if(Nodes.begin(), Nodes.end(), bind2nd(CompareVP, n)) == Nodes.end())
				{
					Queue.push(n);
					Nodes.insert(n);
				}
				else
					delete n;
			}
		}
	}
}
bool isGoal(Node *s)
{
	int goal[4][4] = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 0 }
	};
	return equal((int *)s->state, (int *)s->state + 16, (int *)goal);
}

int main()
{
	// Screen Clear
	system("clear");
	// Time Begin
	clock_t begin = clock();
	// Initialze Puzzle (Random ...)
	Initialize();

	// Print
	cout << "-- Node Generated: 000000";

	int c = 0;
	while (!Queue.empty()) {
		Node* n = Queue.top();
		Queue.pop();
		cout << "\b\b\b\b\b\b" << setw(6) << setfill('0') << c++;
		if (isGoal(n) || MAX_NODE == c) {
			int i = 0;
			cout << endl << endl;
			while (n != NULL) {
				if (i != 0)
					cout << "--Step: " << i++ << endl;
				else
					cout << "--Step: " << i++ << " (Solved)" << endl;
				n->PrintState();
				n = n->ancestor;
			}

			// Time End
			clock_t end = clock();

			// Print
			cout << "-- Total Nodes Generated: " << c << endl;

			// Calculate Time Spent
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			// Print Elapsed Time
			cout.precision(3);
			cout << "-- Elapsed time: " << elapsed_secs << " seconds." << endl << endl;

			if (MAX_NODE == c)
				cout << "-- Sorry, Puzzle not Solvable." << endl << endl;
			else
				cout << "-- Puzzle Solved in " << i - 1 << " Moves." << endl << endl;

			break;
		}
		else
			AddNeighbour(n);		// Solve 15 Puzzle
	}

	for (set<Node*>::iterator p = Nodes.begin();p != Nodes.end();p++)
		delete *p;

	return 0;
}
