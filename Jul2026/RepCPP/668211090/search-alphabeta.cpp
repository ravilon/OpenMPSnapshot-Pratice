/**
 * 
 * Implementation of Alpha/beta pruning
 * 
 * This file contains three different versions
 * 1. Seuential
 * 2. Parallel
 * 3. Parallel with PV splitting
 * 
 * Authors:
 * - Aditya Phopale
 * - Durganshu Mishra
 * - Gaurav Gokhale
 *
 * Original code:
 * (c) 2005, Josef Weidendorfer
 */

#include <stdio.h>
#include <iostream> 
#include "search.h"
#include "board.h"
#include "eval.h"
#include <cstring>

#define MAX_SEARCH 10

class AlphaBetaStrategy : public SearchStrategy {
public:
  AlphaBetaStrategy() : SearchStrategy("AlphaBeta",0) {}
  SearchStrategy *clone() { return new AlphaBetaStrategy(); }


private:
  void searchBestMove();
  /* recursive alpha/beta search */
  int alphabeta(int depth, int alpha, int beta);
  int alphabeta_parallel(int currentdepth, int alpha, int beta, Board& _board, Evaluator& evaluator);
  int alphabeta_pv_split(int currentdepth, int alpha, int beta , int depthOfPv, int curMaxdepth, Board& board, Evaluator& evaluator);
  int alphabeta_transposition(int currentdepth, int alpha, int beta, int depthOfPv, Board& _board, Evaluator& evaluator);
  Variation _pv; 
  bool _inPV;
  bool _foundBestFromPrev;
  
  int _currentMaxDepth; 
  Move _currentBestMove;
  
  int pvAlphaBounds[MAX_SEARCH];
  bool firstPvLeaf=false;
};

/**
 * Entry point for search
 *
 * Does iterative deepening and alpha/beta width handling, and
 * calls alpha/beta search
 */
void AlphaBetaStrategy::searchBestMove() {
  
  firstPvLeaf=false;
    
  int value;
  
  _currentMaxDepth = 0;

  _inPV = (_pv[0].type != Move::none);

  int test;

  omp_set_num_threads(48);
  #pragma omp parallel
  {
     #pragma omp single
      //value = alphabeta_parallel(_currentMaxDepth, -16000, 16000, *_board, *_ev);
      value = alphabeta_pv_split(0, -16000, 16000, 0, SearchStrategy::_maxDepth, *_board, *_ev);
  }
  
  // value = alphabeta(_currentMaxDepth, -16000, 16000);
  _bestMove = _currentBestMove; //update _bestmove

}

/*
 * Alpha/Beta search
 *
 * - first, start with principal variation
 * - depending on depth, we only do depth search for some move types
 */
int AlphaBetaStrategy::alphabeta(int currentdepth, int alpha, int beta) {

    if (currentdepth >= _maxDepth) return evaluate();

    int currentValue = -999999;
    Move m;
    MoveList list;
    generateMoves(list);

    while(list.getNext(m)){
        int value;
        playMove(m); 
        if(currentdepth + 1 < _maxDepth){
            value = -alphabeta(currentdepth+1, -beta, -alpha);
        }
        else{
            value = evaluate();
        }
        takeBack();

        if(value > currentValue){
            currentValue= value;
            foundBestMove(currentdepth, m ,value);

            if (currentdepth == 0) _currentBestMove = m;

        }

        //alpha beta pruning
        if (value > alpha)
        {
            alpha = value;
        }

        if (beta <= alpha)
        {
            break;
        }
    }
    finishedNode(currentdepth, 0);
    return currentValue;
}

int AlphaBetaStrategy::alphabeta_parallel(int currentdepth, int alpha, int beta, Board& board, Evaluator& evaluator) {

  int currentValue = -16000;
  
  Move m;
  MoveList list;
  board.generateMoves(list);

  while(list.getNext(m)){

    bool inParallel = false;

    if(firstPvLeaf && (currentdepth < _maxDepth - 2)){
            inParallel = true;
    }

    if(!inParallel) 
    { 

      board.playMove(m);
      int value;
      if (currentdepth + 1 < _maxDepth)
      {
          value = -alphabeta_parallel(currentdepth + 1, -beta, -alpha, board, evaluator);
      }
      else
      {
          firstPvLeaf = true;
          value = evaluator.calcEvaluation(&board);
      }
      
      board.takeBack();

      if (value > currentValue)
      {
          currentValue= value;
        
          foundBestMove(currentdepth, m, value);
          
          if (currentdepth == 0)
          { 
              _currentBestMove = m;
          }
      }
      if (pvAlphaBounds[currentdepth] > alpha) 
            alpha = pvAlphaBounds[currentdepth];
      // if (((currentdepth % 2) == 0)) 
      // {
          
      // }
      // else
      // {
      // if (-pvAlphaBounds[currentdepth] < beta) 
      //       beta = -pvAlphaBounds[currentdepth];
      // }      
      if (value > alpha) alpha = value;
      
      if (beta <= alpha) break;

    }

    else
    {
      bool get_out = false;
      #pragma omp task firstprivate(m, currentdepth, board, evaluator) shared(currentValue)
      {
        int value;
        board.playMove(m); 

        if(currentdepth + 1 < _maxDepth){
          value = -alphabeta_parallel(currentdepth+1, -beta, -alpha, board, evaluator);
        }

        else{
          value = evaluator.calcEvaluation(&board);
        }

        board.takeBack();

         #pragma omp critical
        {
          if(value > currentValue){
          
            currentValue= value;
            foundBestMove(currentdepth, m ,value);
            if (currentdepth == 0) _currentBestMove = m;
          }

          

          //alpha beta pruning
          if (value > alpha) alpha = value;

          if (beta <= alpha) get_out = true;

          if (pvAlphaBounds[currentdepth] > alpha) 
                alpha = pvAlphaBounds[currentdepth];
          // if (((currentdepth % 2) == 0)) 
          // {
              
          // }
          // else
          // {
          //     if (-pvAlphaBounds[currentdepth] < beta) 
          //       beta = -pvAlphaBounds[currentdepth];
          // }
        }

      }
      if(get_out) break;
    }
    
      
  }
  #pragma omp taskwait 
  return currentValue;
}

int AlphaBetaStrategy::alphabeta_pv_split(int currentdepth, int alpha, int beta , int depthOfPv, int curMaxdepth, Board& board, Evaluator& evaluator){
    
    int currentValue = -16000;

    Move m;
    Move nodeBestMove;
    MoveList list;
    
    //generate moves
    board.generateMoves(list); 

    bool pvNode = !firstPvLeaf;
    if(pvNode)
    {
        pvAlphaBounds[currentdepth] = alpha;
        depthOfPv=currentdepth;
    }
        
    //if we are in the PV, get the next move from the PV
    if(_inPV){
        
        m = _pv[currentdepth];
        
        //if pv move is not in list, set to none
        if(m.type != Move::none && !list.isElement(m,0,true)) 
            m.type = Move::none;
        
        //if no pv move found
        if(m.type == Move::none){ 
            #pragma omp critical
            {
                _inPV = false;
            }
                
        }
    }

    //iterate through each possible move
    while(true) { 

        //if no pv move found, get next from list
        if(m.type == Move::none){ 
            if(!list.getNext(m))
                break;
        }

        bool inParallel = false;

        // PV splitting
        if(pvNode && firstPvLeaf)
            inParallel = true;
        
        // sequential search
        if(!inParallel) { 

            board.playMove(m);
            int value;
            if (currentdepth + 1 < curMaxdepth) 
            {
              value = -alphabeta_pv_split(currentdepth + 1, -beta, -alpha , depthOfPv, curMaxdepth, board, evaluator); 
            }
            else
            {
                firstPvLeaf = true;
                value = evaluator.calcEvaluation(&board);
            }
            board.takeBack();

            if (value > currentValue)
            {
                currentValue = value;

                _pv.update(currentdepth, m);
                foundBestMove(currentdepth, m, value);

                if (currentdepth == 0) _currentBestMove = m;
            }
            if (!pvNode)
            {   
                    
                if ((currentdepth - depthOfPv) % 2 == 0)
                {
                    if (pvAlphaBounds[depthOfPv] > alpha) 
                      alpha = pvAlphaBounds[depthOfPv];
                }
                else
                {
                  if (-pvAlphaBounds[depthOfPv] < beta)  
                    beta = -pvAlphaBounds[depthOfPv];
                }
                
            }

            if (value > alpha) alpha = value;

            if (beta <= alpha) break;

        }
        //parallel search
        else { 

            bool breakLoop = false;
            #pragma omp task firstprivate(m, currentdepth, board, evaluator, depthOfPv) shared(currentValue)
            {   

                board.playMove(m);
                int value;
                if (currentdepth + 1 < curMaxdepth)
                  value = -alphabeta_pv_split(currentdepth + 1, -beta, -alpha , depthOfPv, curMaxdepth, board, evaluator); 
                
                else
                value = evaluator.calcEvaluation(&board);

                board.takeBack();

                #pragma omp critical
                {
                  if (value > currentValue)
                  {
                      currentValue = value;

                      _pv.update(currentdepth, m);

                      foundBestMove(currentdepth, m, value);
 

                      if (currentdepth == 0) _currentBestMove = m;

                  }

                  if (value > alpha)
                  {
                      alpha = value;
                      if (pvNode) pvAlphaBounds[depthOfPv] = value;

                  }

                  if (beta <= alpha) breakLoop = true;

                }
        

                
                if (!pvNode)
                {   

                    if (((currentdepth - depthOfPv) & 1)) 
                    {
                        if (pvAlphaBounds[depthOfPv] > alpha) 
                          alpha = pvAlphaBounds[depthOfPv];

                    }
                    else
                    {
                        if (-pvAlphaBounds[depthOfPv] < beta) 
                          beta = -pvAlphaBounds[depthOfPv];
                    }
                }

            }

            if(breakLoop)
                break;

        }

        m.type = Move::none;
    }
    
    #pragma omp taskwait 
    return currentValue;
}

// Implement the alphabeta pvSplit strategy with transposition table using OpenMP

int AlphaBetaStrategy::alphabeta_transposition(int currentdepth, int alpha, int beta, int depthOfPv, Board& _board, Evaluator& evaluator)
{
  

}










// register ourselve
AlphaBetaStrategy alphabetaStrategy;
