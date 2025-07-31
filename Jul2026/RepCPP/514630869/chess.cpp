#include"chess.h"


const char FIGFILE[] = "images/figures1.png";

 float weights[10][64] = {
    {     -6,-6,-6,-6,-4,-2,6,6,//King
     -8,-8,-8,-8,-6,-4,4,6,
     -8,-8,-8,-8,-6,-4,0,2,
     -10,-10,-10,-10,-8,-4,0,0,
     -10,-10,-10,-10,-8,-4,0,0,
     -8,-8,-8,-8,-6,-4,0,2,
     -8,-8,-8,-8,-6,-4,4,6,
     -6,-6,-6,-6,-4,-2,6,6
     },
     {0,10,2,1,0,1,1,0,//pawn
     0,10,2,1,0,-1,2,0,
      0,10,4,2,0,-2,2,0,
      0,10,6,5,4,0,-4,0,
      0,10,6,5,4,0,-4,0,
      0,10,4,2,0,-2,2,0,
       0,10,2,1,0,-1,2,0,
       0,10,2,1,0,1,1,0
     },

     {-10,-8,-6,-6,-6,-6,-8,-10,//knight
     -8,-4,0,1,0,1,-4,-8,
     -6,0,2,3,3,2,0,-6,
     -6,0,3,4,4,3,1,-6,
    -6,0,3,4,4,3,1,-6,
        -6,0,2,3,3,2,0,-6,
        -8,-4,0,1,0,1,-4,-8,
        -10,-8,-6,-6,-6,-6,-8,-10,
     },
     {-4,-2,-2,-2,-2,-2,-2,-4,//bishop
     -2,0,0,1,0,2,1,-2,
     -2,0,1,1,2,2,0,-2,
     -2,0,2,2,2,2,0,-2,
    -2,0,2,2,2,2,0,-2,
        -2,0,1,1,2,2,0,-2,
        -2,0,0,1,0,2,1,-2,
        -4,-2,-2,-2,-2,-2,-2,-4,

     },{},
     {0,1,1,-1,-1,-1,-1,0,//rook
     0,2,0,0,0,0,0,0,
     0,2,0,0,0,0,0,0,
     0,2,0,0,0,0,0,1,
    0,2,0,0,0,0,0,1,
     0,2,0,0,0,0,0,0,
     0,2,0,0,0,0,0,0,
     0,1,1,-1,-1,-1,-1,0

     },{},{},{},
     {-4,-2,-2,-1,0,-2,-2,-4,//Queen
     -2,0,0,0,0,1,0,-2,
     -2,0,1,1,1,1,1,-2,
     -1,0,1,1,1,1,0,-1,
     -1,0,1,1,1,1,0,-1,
     -2,0,1,1,1,1,1,-2,
     -2,0,0,0,0,1,0,-2,
     -4,-2,-2,-1,0,-2,-2,-4,

     }

 };

int signOf(int n) {
	return -(n < 0) + (n > 0);
}

bool isInArea(int posX, int posY) {
	return posX > -1 && posX<8 && posY > -1 && posY < 8;
}

bool isInArea(int* pos) {
	return isInArea(pos[0], pos[1]);
}

Move::Move() {
	m_oldPos[0] = 0;
	m_oldPos[1] = 0;
	m_newPos[0] = 0;
	m_newPos[1] = 0;
}

Move::Move(int* oldPos, int* newPos) : m_oldPos{ oldPos[0],oldPos[1] }, m_newPos{ newPos[0],newPos[1] }{};

Move::Move(int* oldPos, int newPosX, int newPosY) : m_oldPos{ oldPos[0],oldPos[1] }, m_newPos{ newPosX, newPosY }{};

Move::Move(const Move& obj)
{
	m_oldPos[0] = obj.m_oldPos[0];
	m_oldPos[1] = obj.m_oldPos[1];
	m_newPos[0] = obj.m_newPos[0];
	m_newPos[1] = obj.m_newPos[1];
	//std::cout << "\nCopy constructer\n";
}

Move& Move::operator= (Move const& rhs) {
	if (this != &rhs) {
		m_oldPos[0] = rhs.m_oldPos[0];
		m_oldPos[1] = rhs.m_oldPos[1];
		m_newPos[0] = rhs.m_newPos[0];
		m_newPos[1] = rhs.m_newPos[1];
	}
	return *this;
}

bool Move::operator== (Move const& rhs){
return m_oldPos[0] == rhs.m_oldPos[0] && m_oldPos[1] == rhs.m_oldPos[1] && m_newPos[0] == rhs.m_newPos[0] && m_newPos[1] == rhs.m_newPos[1];
}

Piece::Piece(int color, int posX, int posY) : m_pos{ posX, posY }, m_color(color) { };

bool Piece::isLegalMove(Move _move, ChessBoard& board) {

    std::list<Move> legalMoves;
    addLegalMoves(board,legalMoves);
    for(auto& m : legalMoves){
        if(m == _move) return true;
    }
    return false;

};

void Piece::makeMove(int newPos[2]) {};

int Piece::getColor() { return m_color; };

void Piece::getPos(int& x, int& y) {
		x = m_pos[0];
		y = m_pos[1];
	}


Piece::Piece(const Piece& obj)
{
	m_pos[0] = obj.m_pos[0];
	m_pos[1] = obj.m_pos[1];
	m_color = obj.m_color;
	//std::cout << "\nCopy constructer\n";
}

Piece& Piece::operator= (Piece const& rhs) {
	if (this != &rhs) {
		m_pos[0] = rhs.m_pos[0];
		m_pos[1] = rhs.m_pos[1];
		m_color = rhs.m_color;
	}
	return *this;
}

void Piece::addLegalMoves(ChessBoard& board, std::list<Move>& legalMoves) {
    int* copyb = new int[64];
    board.copyBoard(copyb);
    addLegalMoves(copyb,legalMoves);
    delete[] copyb;
}

void Piece::addLegalMoves(int* board, std::list<Move>& legalMoves){};

Pawn::Pawn(int color, int posX, int posY, bool isFirstMove, bool withSprite) : Piece(color, posX, posY), m_dir(-m_color), m_isFirstMove(isFirstMove) {
    if(withSprite){
            m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*5,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
	m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
    }

};//actually direction depends on playerColor but not on its color

void Pawn::addLegalMoves(int* board, std::list<Move>& legalMoves) {

	int posX = m_pos[0];
	int posY = m_pos[1];
	if (posY == 0 || posY == 7) return;
	if (board[posX*8 + posY + m_dir] == 0) {
		legalMoves.push_back(Move(m_pos, posX, posY + m_dir));
		if (m_isFirstMove && board[posX*8 + posY + 2 * m_dir] == 0) {
			legalMoves.push_back(Move(m_pos, posX, posY + 2 * m_dir));
		}
	}
	if (posX != 7 && board[(posX + 1)*8 + posY + m_dir] * m_color < 0) {
		legalMoves.push_back(Move(m_pos, posX + 1, posY + m_dir));
	}
	if (posX != 0 && board[(posX - 1)*8 + posY + m_dir] * m_color < 0) {
		legalMoves.push_back(Move(m_pos, posX - 1, posY + m_dir));
	}
}

void Pawn::makeMove(int newPos[2]) {
	m_isFirstMove = false;
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}


King::King(int color, int posX, int posY, bool canCastling, bool withSprite) : Piece(color, posX, posY), m_canCastling(canCastling) {
if(withSprite){
        m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*4,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
    m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
}

};

void King::addLegalMoves(int* board, std::list<Move>& legalMoves) {
    int posX = m_pos[0];
    int posY = m_pos[1];
    if(posX>0){

        if (signOf(board[(posX-1)*8 + posY]) != m_color )
            legalMoves.push_back(Move(m_pos, posX-1, posY));

        if(posY>0){
            if (signOf(board[(posX-1)*8 + posY-1]) != m_color )
                legalMoves.push_back(Move(m_pos, posX-1, posY-1));
            if (signOf(board[(posX)*8 + posY-1]) * m_color != 1)
                legalMoves.push_back(Move(m_pos, posX, posY-1));
            }
        if(posY<7){
            if (signOf(board[(posX)*8 + posY+1]) != m_color)
                legalMoves.push_back(Move(m_pos, posX, posY+1));
            if (signOf(board[(posX-1)*8 + posY+1]) != m_color)
                legalMoves.push_back(Move(m_pos, posX-1, posY+1));

            }

    }
    if(posX<7){
        if (signOf(board[(posX+1)*8 + posY]) != m_color)
            legalMoves.push_back(Move(m_pos, posX+1, posY));


        if(posY>0){
            if (signOf(board[(posX+1)*8 + posY-1]) != m_color)
                legalMoves.push_back(Move(m_pos, posX+1, posY-1));
            if (signOf(board[(posX)*8 + posY-1]) != m_color && posX==0)
                legalMoves.push_back(Move(m_pos, posX, posY-1));
        }

        if(posY<7){
            if (signOf(board[(posX+1)*8 + posY+1]) != m_color)
                legalMoves.push_back(Move(m_pos, posX+1, posY+1));
            if (signOf(board[(posX)*8 + posY+1]) != m_color && posX==0)
                legalMoves.push_back(Move(m_pos, posX, posY+1));
        }


    }


	if (m_canCastling) {
			if (board[(m_pos[0] + 1)*8 + m_pos[1]] == 0 && board[(m_pos[0] + 2)*8 + m_pos[1]] == 0){
				legalMoves.push_back(Move(m_pos, m_pos[0] + 2, m_pos[1]));
			}
//			if (board.getCell(m_pos[0] - 1, m_pos[1]) == 0 && board.getCell(m_pos[0] - 2, m_pos[1]) == 0
//				&& board.getCell(m_pos[0] - 3, m_pos[1]) == 0 && board.canCastling(m_color, 0)) {
//				legalMoves.push_back(Move(m_pos, m_pos[0] - 2, m_pos[1]));
//			}
//
	}


}

void King::makeMove(int newPos[2]) {
	m_canCastling = false;
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}

Rook::Rook(int color, int posX, int posY, bool withSprite) : Piece(color, posX, posY){

if(withSprite){
       m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*0,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
    m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
}
};


void Rook::addLegalMoves(int* board, std::list<Move>& legalMoves) {

    int posX = m_pos[0];
    int posY = m_pos[1];

    int d = 1;
    while(board[(posX-d)*8+posY]==0 && posX-d>-1){
        legalMoves.push_back(Move(m_pos, posX-d, posY));
        ++d;
    }
    if(posX-d>-1 && signOf(board[(posX-d)*8+posY])!= m_color)
        legalMoves.push_back(Move(m_pos, posX-d, posY));


    d=1;
    while(board[posX*8+posY-d]==0 && posY-d>-1){
        legalMoves.push_back(Move(m_pos, posX, posY-d));
        ++d;
    }
    if(posY-d>-1 && signOf(board[posX*8+posY-d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX, posY-d));


    d=1;
    while(board[posX*8+posY+d]==0 && posY+d<8){
        legalMoves.push_back(Move(m_pos, posX, posY+d));
        ++d;
    }
    if(posY+d<8 && signOf(board[posX*8+posY+d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX, posY+d));

    d=1;
    while(board[(posX+d)*8+posY]==0 && posX+d<8){
        legalMoves.push_back(Move(m_pos, posX+d, posY));
        ++d;
    }
    if(posX+d<8 && signOf(board[(posX+d)*8+posY])!= m_color)
        legalMoves.push_back(Move(m_pos, posX+d, posY));

}


void Rook::makeMove(int newPos[2]) {
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}

Bishop::Bishop(int color, int posX, int posY, bool withSprite) : Piece(color, posX, posY){

if(withSprite){
       m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*2,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
    m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
}


};

void Bishop::addLegalMoves(int* board, std::list<Move>& legalMoves) {

    int posX = m_pos[0];
    int posY = m_pos[1];

    int d = 1;
    while(board[(posX-d)*8+posY-d]==0 && posX-d>-1 && posY-d>-1){
        legalMoves.push_back(Move(m_pos, posX-d, posY-d));
        ++d;
    }
    if(posX-d>-1 && posY-d>-1&& signOf(board[(posX-d)*8+posY-d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX-d, posY-d));


    d=1;
    while(board[(posX+d)*8+posY-d]==0 && posY-d>-1 && posX+d<8){
        legalMoves.push_back(Move(m_pos, posX+d, posY-d));
        ++d;
    }
    if(posY-d>-1 && posX+d<8 && signOf(board[(posX+d)*8+posY-d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX+d, posY-d));


    d=1;
    while(board[(posX-d)*8+posY+d]==0 && posY+d<8 && posX-d>-1){
        legalMoves.push_back(Move(m_pos, posX-d, posY+d));
        ++d;
    }
    if(posY+d<8 && posX-d >-1 && signOf(board[(posX-d)*8+posY+d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX-d, posY+d));

    d=1;
    while(board[(posX+d)*8+posY+d]==0 && posX+d<8 && posY+d<8){
        legalMoves.push_back(Move(m_pos, posX+d, posY+d));
        ++d;
    }
    if(posX+d<8 && posY+d<8 && signOf(board[(posX+d)*8+posY+d])!= m_color)
        legalMoves.push_back(Move(m_pos, posX+d, posY+d));




}

void Bishop::makeMove(int newPos[2]) {
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}


Knight::Knight(int color, int posX, int posY, bool withSprite) : Piece(color, posX, posY) {
if(withSprite){
        m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*1,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
    m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
}

};

void Knight::addLegalMoves(int* board, std::list<Move>& legalMoves) {

	int dx[4] = { -2, -1, 1, 2 };
	int dy[2] = { -1, 1 };
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 2; ++j) {
			int posX = m_pos[0] + dx[i];
			int posY = m_pos[1] + (3 - abs(dx[i])) * dy[j];
			if (isInArea(posX, posY)) {

				if (signOf(board[posX*8+ posY]) * m_color != 1) {
					legalMoves.push_back(Move(m_pos, posX, posY));

				}
			}

		}
	}
}

void Knight::makeMove(int newPos[2]) {
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}

Queen::Queen(int color, int posX, int posY, bool withSprite) : Piece(color, posX, posY) {

if(withSprite){
        m_texture.loadFromFile(FIGFILE);
    m_sprite.setTexture(m_texture);
    int y = color>0? 1:0;
	m_sprite.setTextureRect(IntRect(SIZE_OF_CELL*3,SIZE_OF_CELL*y,SIZE_OF_CELL,SIZE_OF_CELL));
    m_sprite.setPosition(SIZE_OF_CELL*posX,SIZE_OF_CELL*posY);
}

};

void Queen::addLegalMoves(int* board, std::list<Move>& legalMoves) {
    Bishop(m_color,m_pos[0],m_pos[1]).addLegalMoves(board,legalMoves);
    Rook(m_color,m_pos[0],m_pos[1]).addLegalMoves(board,legalMoves);
}

void Queen::makeMove(int newPos[2]) {
	m_pos[0] = newPos[0];
	m_pos[1] = newPos[1];
}


ChessBoard::ChessBoard(int playerColor) : m_board{ -ROOK, -PAWN, 0 , 0, 0, 0, PAWN,ROOK,
												-KNIGHT, -PAWN, 0 , 0, 0, 0, PAWN,KNIGHT,
												-BISHOP, -PAWN, 0 , 0, 0, 0, PAWN,BISHOP,
												-QUEEN, -PAWN, 0 , 0, 0, 0, PAWN,QUEEN,
												-KING, -PAWN, 0 , 0, 0, 0, PAWN,KING,
												-BISHOP, -PAWN, 0 , 0, 0, 0, PAWN,BISHOP,
												-KNIGHT, -PAWN, 0 , 0, 0, 0, PAWN,KNIGHT,
												-ROOK, -PAWN, 0 , 0, 0, 0, PAWN,ROOK },
												m_playerColor(playerColor)
{
	if (m_playerColor == BLACK) {
		for (int i = 0; i < 64; ++i) {
			m_board[i] *= -1;
		}
	}

	//Creating GameObjects
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			int piece = m_board[i*8+j];
			if (piece != 0) {
				switch (abs(piece)) {
				case PAWN:
					piecesOnBoard.push_back(new Pawn(signOf(piece),i, j));
					break;
				case KNIGHT:
					piecesOnBoard.push_back(new Knight(signOf(piece), i, j));
					break;
				case KING:
					piecesOnBoard.push_back(new King(signOf(piece), i, j));
					break;
				case ROOK:
					piecesOnBoard.push_back(new Rook(signOf(piece), i, j));
					break;
				case BISHOP:
					piecesOnBoard.push_back(new Bishop(signOf(piece), i, j));
					break;
				case QUEEN:
					piecesOnBoard.push_back(new Queen(signOf(piece), i, j));
					break;
				}

			}
			else {
				piecesOnBoard.push_back(new Piece(0, 0, 0));
			}
		}

	}
};

	int ChessBoard::getCell(int posX, int posY) {
		return m_board[8 * posX + posY];
	}
	int ChessBoard::getCell(int pos[2]) {
		return getCell(pos[0], pos[1]);
	}

	void ChessBoard::setCell(int posX, int posY, int val) {
		m_board[8 * posX + posY] = val;
	}
	void ChessBoard::setCell(int pos[2], int val) {
		setCell(pos[0], pos[1], val);
	}

	void ChessBoard::makeMove(Move move) {
		makeMove(move.m_oldPos, move.m_newPos);
	}
	void ChessBoard::makeMove(int oldPos[2], int newPos[2]) {

		piecesOnBoard.at(newPos[0] * 8 + newPos[1]) = piecesOnBoard.at(oldPos[0] * 8 + oldPos[1]);
		piecesOnBoard.at(newPos[0] * 8 + newPos[1])->makeMove(newPos);
		piecesOnBoard.at(oldPos[0] * 8 + oldPos[1]) = new Piece(0,0,0);
		int piece = getCell(oldPos);
		setCell(oldPos, 0);
		setCell(newPos, piece);


		if (abs(piece) == KING && abs(oldPos[0] - newPos[0]) == 2) {
            piecesOnBoard.at(5 * 8 + newPos[1]) = piecesOnBoard.at(7 * 8 + oldPos[1]);
            std::cout<<"CASTLING";
            int rook_pos[2] = {5,newPos[1]};
            piecesOnBoard.at(5 * 8 + newPos[1])->makeMove(rook_pos);
            piecesOnBoard.at(7 * 8 + oldPos[1]) = new Piece(0,0,0);
            setCell(rook_pos, signOf(piece)*ROOK);
            rook_pos[0] = 7;
            setCell(rook_pos, 0);

		}

	};

	void ChessBoard::show() {
		std::cout << "\n";
		for (int i = 0; i <8 ; ++i) {
			for (int j = 0; j < 8; ++j) {
				std::cout << m_board[8 * j + i] << "\t";
			}
			std::cout << "\n";
		}

		std::cout << "\n";
	}


	int ChessBoard::getPlayerColor(){
	return m_playerColor;
	}

    void ChessBoard::copyBoard(int* copy_b){

    memcpy(copy_b, m_board, 64 * sizeof(int));
	}

	void ChessBoard::loadPosition()
{

    for (auto& piece : piecesOnBoard) {
        int x,y;
            piece->getPos(x,y);
            piece->m_sprite.setPosition(SIZE_OF_CELL*x,SIZE_OF_CELL*y);
    }
}


	void ChessAI::findAllLegalMoves(ChessBoard& board, std::list<Move>& allLegalMoves) {
		for (auto& piece : board.piecesOnBoard) {
			if (piece->getColor() * m_AIColor == 1) {
				piece->addLegalMoves(board, allLegalMoves);
			}
		}
	}

    void ChessAI::findAllLegalMoves(int* board, std::list<Move>& allLegalMoves, bool isPlayer) {

        int neededColor = (1 - 2*isPlayer)*m_AIColor;
		for(int i=0;i<8;++i){
           for(int j=0;j<8;++j){
                int color = signOf(board[i*8+j]);
            if( color == neededColor){
                switch(abs(board[i*8+j])){
                case PAWN:
					Pawn(color,i, j,false).addLegalMoves(board, allLegalMoves);
					break;
				case KNIGHT:
					Knight(color, i, j,false).addLegalMoves(board, allLegalMoves);
					break;
				case ROOK:
					Rook(color, i, j,false).addLegalMoves(board, allLegalMoves);
					break;
				case BISHOP:
					Bishop(color, i, j,false).addLegalMoves(board, allLegalMoves);
					break;
				case QUEEN:
					Queen(color, i, j,false).addLegalMoves(board, allLegalMoves);
					break;
				case KING:
					King(color, i, j,false).addLegalMoves(board, allLegalMoves);
					break;
                }
            }
           }
		}

	}

	ChessAI::ChessAI(int depthOfAnalysis, int AIColor) : m_depthOfAnalysis(depthOfAnalysis), m_AIColor(AIColor) {
        int size = m_numberOfMovesInTable*(m_depthOfAnalysis+1);
        m_bestMovesTable = new Move[size];
//        for(int i=0;i<size;++i){
//            m_bestMovesTable[i] = Move;
//        }
	}

	ChessAI::~ChessAI(){
	delete[] m_bestMovesTable;
	}

	int ChessAI::estimatePosition(int* board){

        int score = 0;
		for (int i = 0; i < 64; ++i) {
            int piece = board[i];
			if (piece != 0) {
                    int n=(abs(piece)/20)%90;
                    int k = signOf(piece)>0 ? (i) : ((i/8)*8+ (7-i%8));
				score += piece + 1*signOf(piece)*weights[n][k];
			}
		}

		return score;
	}

    void ChessBoard::showWithPos() {
		std::cout << "\n";
		for (int i = 0; i <8 ; ++i) {
			for (int j = 0; j < 8; ++j) {
                int piece = m_board[8 * j + i];
                if (piece != 0) {
                        int p = 8 * j + i;
                    int n=(abs(piece)/20)%90;
                    int k = signOf(piece)>0 ? (p) : ((p/8)*8+ (7-p%8));
                    std::cout << piece + 1*signOf(piece)*weights[n][k]<< "\t";
                }
				else{
                    std::cout << piece << "\t";
				}
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}


	void ChessAI::undo(int* board, Move move, int attackedPiece){

	int newPos = move.m_newPos[0]*8+move.m_newPos[1];
	int piece = board[move.m_oldPos[0]*8+move.m_oldPos[1]] = board[newPos];
	board[newPos] = attackedPiece;


    if(abs(piece)==KING && move.m_newPos[0]-move.m_oldPos[0] == 2){
        board[7*8+move.m_oldPos[1]] =ROOK * signOf(piece);
        board[5*8+move.m_oldPos[1]] = 0;
    }
	}


	void ChessAI::makeMove(Move move, int* board){
        int oldPos = move.m_oldPos[0]*8+move.m_oldPos[1];
        int piece = board[move.m_newPos[0]*8+move.m_newPos[1]] = board[oldPos];
        board[oldPos] = 0;
        if(abs(piece)==KING && move.m_newPos[0]-move.m_oldPos[0] == 2){
            board[7*8+move.m_oldPos[1]] =0;
            board[5*8+move.m_oldPos[1]] = ROOK * signOf(piece);
        }
	}



    Move ChessAI::findBestMove5(ChessBoard& board) {

        int* new_board = new int[64];
        board.copyBoard(new_board);

		std::list<Move> allLegalMoves;
        int numberOfCalcMoves = 0;
		findAllLegalMoves(board, allLegalMoves);

        int min_score = 9999;
        Move bestMove;
		for (auto& move : allLegalMoves) {
            int attackedPiece = board.getCell(move.m_newPos);
            makeMove(move, new_board);
            int score = minimaxAB(new_board,m_depthOfAnalysis,true,100,numberOfCalcMoves);
            if (score<min_score){
                bestMove = move;
                min_score = score;
            }
            undo(new_board,move,attackedPiece);
		}
        std::cout<<"\n numberOfCalcMoves5 = "<<numberOfCalcMoves<<"\n";
		delete[] new_board;
		return bestMove;
	};

	int ChessAI::minimaxAB(int* board, int depth, bool isPlayer, int estimate, int& numberOfCalcMoves){

        if(depth==0){
            return estimatePosition(board);
        }

		std::list<Move> allLegalMoves;

		findAllLegalMoves(board, allLegalMoves,isPlayer);

		if(isPlayer){
            int max_score = -9999;
            for (auto& move : allLegalMoves) {
                numberOfCalcMoves++;
                int attackedPiece = board[move.m_newPos[0]*8+move.m_newPos[1]];
                makeMove(move, board);
                int score = minimaxAB(board,depth - 1,!isPlayer,max_score,numberOfCalcMoves);
                undo(board,move,attackedPiece);
                if (score>max_score){
                    if(score> estimate){
                        return score;
                    }
                max_score = score;
                }

            }
            return max_score;
		}
		else{
            int min_score = 9999;
            for (auto& move : allLegalMoves) {
                numberOfCalcMoves++;
                int attackedPiece = board[move.m_newPos[0]*8+move.m_newPos[1]];
                makeMove(move, board);
                int score = minimaxAB(board,depth - 1,!isPlayer,min_score,numberOfCalcMoves);
                undo(board,move,attackedPiece);
                if (score<min_score){
                    if(score < estimate){
                        return score;
                    }
                min_score = score;
                }

            }
            return min_score;
		}

	}

    Move ChessAI::findBestMoveParallel(ChessBoard& board) {

        int np = 1;
        omp_set_num_threads(np);
        int* new_board = new int[64*np];
        for(int i=0;i<np;++i)
            board.copyBoard(new_board+i*64);


		std::list<Move> allLegalMoves;
        int numberOfCalcMoves = 0;
		findAllLegalMoves(board, allLegalMoves);
        int size = allLegalMoves.size();
        int min_score = 9999;
        Move bestMove;

        #pragma omp parallel shared(board,new_board,min_score,bestMove,allLegalMoves)
        {
            int id = omp_get_thread_num();
            int len = size/np;
            auto beg = allLegalMoves.begin();
            for(int i=0;i<id*len;++i) beg++;
            auto end = beg;
            for(int i=0;i<len;++i) end++;
            if(id==np-1) end = allLegalMoves.end();
            for (auto& it = beg;it!=end;++it) {
                Move move = *it;
                int attackedPiece = board.getCell(move.m_newPos);

                makeMove(move, new_board+id*64);
                int score = minimaxAB(new_board+id*64,m_depthOfAnalysis,true,20000,numberOfCalcMoves);
            #pragma omp critical
                if (score<min_score){
                    bestMove = move;
                    min_score = score;
                    }
                undo(new_board+id*64,move,attackedPiece);
            }
        }





		//if (allLegalMoves.size() == 0) return Move();
        std::cout<<"\n numberOfCalcMoves5 = "<<numberOfCalcMoves<<"\n";
		delete[] new_board;
		return bestMove;
	};


	void ChessAI::addBestMovesToTable(std::list<Move>& bestMoves, int depth){

        auto it = bestMoves.begin();
        for(int i=0;i<m_numberOfMovesInTable;++i){
#pragma omp critical
            m_bestMovesTable[depth*m_numberOfMovesInTable+i] = *it;
            ++it;
        }

	}


	int ChessAI::minimaxABKiller(int* board, int depth, bool isPlayer, int estimate, int& numberOfCalcMoves){

        if(depth==0){
            return estimatePosition(board);
        }

		std::list<Move> allLegalMoves;

		findAllLegalMoves(board, allLegalMoves,isPlayer);

        for(auto move = allLegalMoves.begin();move!=allLegalMoves.end();++move){
           auto it = allLegalMoves.begin();
           for(int i=0;i<m_numberOfMovesInTable;++i){
               ++it;
                if(*move == m_bestMovesTable[depth*m_numberOfMovesInTable+i]){

                    std::swap(*move,*it);
                    break;
                }
           }
        }
//        int i=0;int j=0;
//        for(auto& move1 : allLegalMoves){
//            for(auto& move2 : allLegalMoves){
//
//                if(move1==move2 && j!=i){
//                    std::cout<<"fuck";
//                }
//                j++;
//            }
//            i++;
//            j=0;
//        }
        std::list<Move> bestMoves;
		if(isPlayer){
            int max_score = -9999;

		for (auto& move : allLegalMoves) {
            numberOfCalcMoves++;
            int attackedPiece = board[move.m_newPos[0]*8+move.m_newPos[1]];
            makeMove(move, board);
            int score = minimaxAB(board,depth - 1,!isPlayer,max_score,numberOfCalcMoves);
            undo(board,move,attackedPiece);
            if (score>max_score){
                    bestMoves.push_front(move);
                    if(score> estimate){
                        addBestMovesToTable(bestMoves,depth);
                        return score;
                    }
                max_score = score;
            }

		}
            addBestMovesToTable(bestMoves,depth);
            return max_score;
		}
		else{
            int min_score = 9999;
		for (auto& move : allLegalMoves) {
		    numberOfCalcMoves++;
            int attackedPiece = board[move.m_newPos[0]*8+move.m_newPos[1]];
            makeMove(move, board);
            int score = minimaxAB(board,depth - 1,!isPlayer,min_score,numberOfCalcMoves);
            undo(board,move,attackedPiece);
            if (score<min_score){
                bestMoves.push_front(move);
                if(score < estimate){
                    addBestMovesToTable(bestMoves,depth);
                    return score;
                }
                min_score = score;
            }

		}
		addBestMovesToTable(bestMoves,depth);
		return min_score;
		}

	}




Move ChessAI::findBestMoveKiller(ChessBoard& board) {
int np = 4;
omp_set_num_threads(np);
        int* new_board = new int[64*np];
        for(int i=0;i<np;++i)
            board.copyBoard(new_board+i*64);

		std::list<Move> allLegalMoves;
        int numberOfCalcMoves = 0;
		findAllLegalMoves(board, allLegalMoves);//parallel
        int size = allLegalMoves.size();
        int min_score = 9999;
        Move bestMove;

        #pragma omp parallel shared(board,new_board,min_score,bestMove,allLegalMoves)
        {
            int id = omp_get_thread_num();
            int len = size/np;
            //std::cout<<"id="<<size<<"\n";
            auto beg = allLegalMoves.begin();
            for(int i=0;i<id*len;++i) beg++;
            auto end = beg;
            for(int i=0;i<len;++i) end++;
            if(id==np-1) end = allLegalMoves.end();
            for (auto& it = beg;it!=end;it++) {
                Move move = *it;
                int attackedPiece = board.getCell(move.m_newPos);

                makeMove(move, new_board+id*64);
                int score = minimaxABKiller(new_board+id*64,m_depthOfAnalysis,true,20000,numberOfCalcMoves);
            #pragma omp critical
                if (score<min_score){
                    bestMove = move;
                    min_score = score;
                    }
                undo(new_board+id*64,move,attackedPiece);
            }
        }





		//if (allLegalMoves.size() == 0) return Move();
        std::cout<<"\n numberOfCalcMoves5 = "<<numberOfCalcMoves<<"\n";
		delete[] new_board;
		return bestMove;
	};

