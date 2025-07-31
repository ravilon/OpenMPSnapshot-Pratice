#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <list>
#include <cstdlib>
#include <iterator>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <omp.h>

typedef double T;

using namespace sf;

static int SIZE_OF_CELL = 60;

#define  MIN(a,b) (((a)<(b)) ? (a) : (b))

#define WHITE 1
#define BLACK -1



int signOf(int n);

class ChessBoard;

enum PIECES_NAMES {//and their weights
	PAWN = 20,
	KNIGHT = 55,
	BISHOP = 60,
	ROOK = 100,
	QUEEN = 180,
	KING = 1800
};

class Move {
public:
	int m_oldPos[2];
	int m_newPos[2];
	Move();
	Move(int* oldPos, int* newPos);
	Move(int* oldPos, int newPosX, int newPosY);
	Move(const Move& obj);

	Move& operator= (Move const& rhs);

	bool operator== (Move const& rhs);
	void show() {
		std::cout << "{ " << m_oldPos[0] << ", " << m_oldPos[1] << " }, { "<< m_newPos[0] << ", " << m_newPos[1] << " } \n";
	}
};

class Piece {
protected:
	//GameObject
	int m_pos[2];//position on board
	int m_color;

public:
    Sprite m_sprite;
    Texture m_texture;
	Piece(int color = WHITE, int posX = 0, int posY = 0);
	virtual bool isLegalMove(Move _move, ChessBoard& board);
	void addLegalMoves(ChessBoard& board, std::list<Move>& legalMoves);
	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves);
	virtual void makeMove(int newPos[2]);
	int getColor();
	void getPos(int& x, int& y);
	Piece(const Piece& obj);
	Piece& operator= (Piece const& rhs);
};

class Pawn : public Piece {
private:
	bool m_isFirstMove = true;
	int m_dir;
public:
	Pawn(int color = WHITE, int posX = 0, int posY = 0, bool isFirstMove = true, bool withSprite = true);

	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};

class Knight : public Piece {

public:
	Knight(int color = WHITE, int posX = 0, int posY = 0, bool withSprite = true);

	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};

class King : public Piece {
private:
	bool m_canCastling = true;
	bool m_isAttacked = false;

public:
	King(int color = WHITE, int posX = 0, int posY = 0, bool canCastling = true, bool withSprite = true);


	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};

class Rook : public Piece {

public:
	Rook(int color = WHITE, int posX = 0, int posY = 0, bool withSprite = true);

	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};

class Bishop : public Piece {

public:
	Bishop(int color = WHITE, int posX = 0, int posY = 0, bool withSprite = true);

	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};

class Queen : public Piece {

public:
	Queen(int color = WHITE, int posX = 0, int posY = 0, bool withSprite = true);

	virtual void addLegalMoves(int* board, std::list<Move>& legalMoves) override;
	virtual void makeMove(int newPos[2]) override;
};


class ChessBoard {
private:
	int m_playerColor;
    int m_board[64];

public:
    std::vector<Piece*> piecesOnBoard;
	ChessBoard(int playerColor = WHITE);

    int getPlayerColor();
	void copyBoard(int* copy_b);
	bool canCastling(int color, int dir);
	int getCell(int posX, int posY);
	int getCell(int pos[2]);

	void setCell(int posX, int posY, int val);
	void setCell(int pos[2], int val);

	void makeMove(Move move);
	void makeMove(int oldPos[2], int newPos[2]);

	void show();
    void showWithPos();
	void loadPosition();

};


class ChessAI {
private:
	int m_depthOfAnalysis;
	int m_AIColor;
	int m_numberOfMovesInTable = 2;
    Move* m_bestMovesTable;

	void findAllLegalMoves(ChessBoard& board, std::list<Move>& allLegalMoves);
	void findAllLegalMoves(int* board, std::list<Move>& allLegalMoves, bool isPlayer = false);
public:
	ChessAI(int depthOfAnalysis, int AIColor = BLACK);
	~ChessAI();
    void undo(int* board, Move move, int attackedPiece);
	void makeMove(Move _move, int* board);
	int minimaxAB(int* board, int depth, bool isPlayer, int estimate, int& numberOfCalcMoves);
    int minimaxABKiller(int* board, int depth, bool isPlayer, int estimate, int& numberOfCalcMoves);
	Move findBestMove5(ChessBoard& board);
	Move findBestMoveParallel(ChessBoard& board);
	Move findBestMoveKiller(ChessBoard& board);
	void addBestMovesToTable(std::list<Move>& bestMoves, int depth);
    int estimatePosition(int* board);
};

