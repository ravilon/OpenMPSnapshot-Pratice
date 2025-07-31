/**
 * A real world, parallel strategy:
 * Alpha/Beta with Iterative Deepening (ABID)
 *
 * Original sequential strategy: 2005, Josef Weidendorfer
 * Parallel implementation: Aditya Phopale, Durganshu Mishra and Gaurav Gokhale
 */

#include <stdio.h>
#include <iostream>
#include "search.h"
#include "board.h"
#include "eval.h"
#include <omp.h>

class ABIDStrategy: public SearchStrategy
{
 public:
    ABIDStrategy(): SearchStrategy("ABID", 2) {}
    SearchStrategy* clone() { return new ABIDStrategy(); }

    Move& nextMove() { return _pv[1]; }

 private:
    void searchBestMove();
    /* recursive alpha/beta search */
    int alphabeta(int depth, int alpha, int beta, Board& board, Evaluator& ev);

    /* prinicipal variation found in last search */
    Variation _pv;
    Move _currentBestMove;
    bool _inPV;
    int _currentMaxDepth;
	omp_lock_t lockArray[10];
};


/**
 * Entry point for search
 *
 * Does iterative deepening and alpha/beta width handling, and
 * calls alpha/beta search
 */
void ABIDStrategy::searchBestMove()
{    
    int alpha = -15000, beta = 15000;
    int nalpha, nbeta, currentValue = 0;
	for (int i=0; i<10; i++)
        omp_init_lock(&(lockArray[i]));

    _pv.clear(_maxDepth);
    _currentBestMove.type = Move::none;
    _currentMaxDepth=1;
    
    /* iterative deepening loop */
    do {

	/* searches on same level with different alpha/beta windows */
	while(1) {

	    nalpha = alpha, nbeta = beta;
	    _inPV = (_pv[0].type != Move::none);

	    if (_sc && _sc->verbose()) {
		char tmp[100];
		sprintf(tmp, "Alpha/Beta [%d;%d] with max depth %d", alpha, beta, _currentMaxDepth);
		_sc->substart(tmp);
	    }
// #pragma omp parallel
// {
// 		std::cout<<"Hey    "<<'\n';
// }

#pragma omp parallel
{
#pragma omp single
	    currentValue = alphabeta(0, alpha, beta, *_board, *_ev);
}
	    /* stop searching if a win position is found */
	    if (currentValue > 14900 || currentValue < -14900)
		_stopSearch = true;

	    /* Don't break out if we haven't found a move */
	    if (_currentBestMove.type == Move::none)
		_stopSearch = false;

	    if (_stopSearch) break;

	    /* if result is outside of current alpha/beta window,
	     * the search has to be rerun with widened alpha/beta
	     */
	    if (currentValue <= nalpha) {
		alpha = -15000;
		if (beta<15000) beta = currentValue+1;
		continue;
	    }
	    if (currentValue >= nbeta) {
		if (alpha > -15000) alpha = currentValue-1;
		beta=15000;
		continue;
	    }
	    break;
	}

	/* Window in both directions cause of deepening */
	alpha = currentValue - 200, beta = currentValue + 200;

	if (_stopSearch) break;

	_currentMaxDepth++;
    }
    while(_currentMaxDepth <= _maxDepth);

    _bestMove = _currentBestMove;
	for (int i=0; i<10; i++)
        omp_destroy_lock(&(lockArray[i]));
}


/*
 * Alpha/Beta search
 *
 * - first, start with principal variation
 * - depending on depth, we only do depth search for some move types
 */
int ABIDStrategy::alphabeta(int depth, int alpha, int beta, Board& board, Evaluator& evaluator)
{

    // int currentValue = -14999+depth, value;
	int value;
	int someVal = -15999;
  	int* currentValue = &someVal;
    Move m;
    MoveList list;
    bool depthPhase, doDepthSearch;

    /* We make a depth search for the following move types... */
    int maxType = (depth < _currentMaxDepth-1)  ? Move::maxMoveType :
	          (depth < _currentMaxDepth)    ? Move::maxPushType :
	                                          Move::maxOutType;
	
    board.generateMoves(list);
	
    // if (_sc && _sc->verbose()) {
	//     char tmp[100];
	//     sprintf(tmp, "Alpha/Beta [%d;%d], %d moves (%d depth)", alpha, beta,
	// 	    list.count(Move::none), list.count(maxType));
	//     _sc->startedNode(depth, tmp);
    // }

    /* check for an old best move in principal variation */
    if (_inPV) {
		m = _pv[depth];

		if ((m.type != Move::none) && (!list.isElement(m, 0, true)))
	    	m.type = Move::none;

		if (m.type == Move::none){ 
#pragma omp critical (update_inPv)
{
			_inPV = false;
}
		}
	}

    // first, play all moves with depth search
    depthPhase = true;

    while (1) {

	bool returnvalue = false;
	// get next move
	if (m.type == Move::none) {
            if (depthPhase)
		depthPhase = list.getNext(m, maxType);
            if (!depthPhase)
		if (!list.getNext(m, Move::none)) break;
	}
	// we could start with a non-depth move from principal variation
	doDepthSearch = depthPhase && (m.type <= maxType);
	
#pragma omp task firstprivate(m, depth, board, currentValue, evaluator)
{	
	board.playMove(m);

	/* check for a win position first */
	if (!board.isValid()) {

	    /* Shorter path to win position is better */
	    value = 14999-depth;
	}
	else {

            if (doDepthSearch) {
				
		/* opponent searches for its maximum; but we want the
		 * minimum: so change sign (for alpha/beta window too!)
		 */
		value = -alphabeta(depth+1, -beta, -alpha, board, evaluator);
            }
            else {
		value = evaluator.calcEvaluation(&board);
	    }
	}
	
	board.takeBack();
	
	omp_set_lock(&(lockArray[depth]));
	/* best move so far? */
	if (value > *currentValue) {
	    *currentValue = value;
	    _pv.update(depth, m);
		
	    if (_sc) _sc->foundBestMove(depth, m, *currentValue);
	    if (depth == 0)
		    _currentBestMove = m;
		
	    /* alpha/beta cut off or win position ... */
	    if (*currentValue>15900 || *currentValue >= beta) {
		if (_sc) _sc->finishedNode(depth, _pv.chain(depth));
			returnvalue = true;
	    }
		
	    /* maximize alpha */
	    if (*currentValue > alpha) alpha = *currentValue;
	}
	omp_unset_lock(&(lockArray[depth]));
}
	if(returnvalue){
		return *currentValue;
	}

	if (_stopSearch) break; // depthPhase=false;
	m.type = Move::none;
    }
#pragma omp taskwait
    if (_sc) _sc->finishedNode(depth, _pv.chain(depth));
	
    return *currentValue;
}

// register ourselve
ABIDStrategy abidStrategy;
