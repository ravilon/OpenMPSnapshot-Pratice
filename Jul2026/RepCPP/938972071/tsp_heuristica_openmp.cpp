/*
 * COPPE/UFRJ (15/ABR/2021)
 * COS760 - Arquiteruas Avancadas
 * 
 * 001-TSP
 * 
 * Nome: Luiz Marcio Faria de Aquino Viana
 * DRE: 120048833
 * CPF: 024.723.347-10
 * RG: 08855128-8 IFP-RJ
 * 
 * tsp_heuristica_openmp.cpp
 */

#include"tsp_all.h"

CTSPHeuristicaOpenMP::CTSPHeuristicaOpenMP(
    int appid, 
    int nprocs, 
    int ftype, 
    datatest_t* result, 
    datatokenptr_t* datawrk, 
    datatokenptr_t* datares, 
    long datasz)
{
    m_result = result; 

    m_ftype = ftype;

    m_datawrk = datawrk;
    m_datares = datares;
    m_datasz = datasz;

    m_appid = appid;
    m_nprocs = nprocs;
}

CTSPHeuristicaOpenMP::~CTSPHeuristicaOpenMP()
{
    //TODO:
}

/* Methodes */

long CTSPHeuristicaOpenMP::findItemAt(long currpos, long* tidStartData)
{
    for(long j = 1; j < DEF_NUM_OF_THREADS; j++)
    {
        long pos0 = tidStartData[j] - 1;
        long pos1 = tidStartData[j];
        if((currpos == pos0) || (currpos == pos1))
            return TRUE;
    }
    return FALSE;
}

double CTSPHeuristicaOpenMP::calculateMinCost(datatokenptr_t* p1, datatokenptr_t* p2, datatokenptr_t* datawrk, long datasz, long* tidStartData)
{
    double min_cost = DEF_MAX_VALUE;

    datatoken_t* p2_dataptr = NULL;

    for(long i = 1; i < datasz; i++) 
    {
        long bFound = findItemAt(i, tidStartData);
        if(bFound == TRUE) 
        {
            p2_dataptr = p2->initial_dataptr;
        
            double d = calculateEuclideanDistance(m_ftype, p1->dataptr, p2->initial_dataptr);
            if(d < min_cost)
                min_cost = d;
        }
        else 
        {
            datatokenptr_t* p_tmp = & datawrk[i];

            if(p_tmp->dataptr->datatoken_processed == FALSE) {
                double d = calculateEuclideanDistance(m_ftype, p1->dataptr, p_tmp->initial_dataptr);
                if(d < min_cost) {
                    p2_dataptr = p_tmp->initial_dataptr;
                    min_cost = d;
                }
            }
        }
    }

#pragma omp critical
    {
        if(p2_dataptr != NULL) {
            p2->dataptr = p2_dataptr;
            p2->dataptr->datatoken_processed = TRUE;
            p2->datatoken_dist = min_cost;
        }
        else {
            min_cost = 0;
        }
    }

    return min_cost;
}

double CTSPHeuristicaOpenMP::calculateTotalCost(datatokenptr_t* data, long datasz)
{
    datatokenptr_t* p0;
    datatokenptr_t* p1;

    double ctotal, d;
    long k;
    
    ctotal = 0.0;

    p0 = & data[0];
    p0->datatoken_dist = 0.0;
    for(k = 1; k < datasz; k++) 
    {
        p1 = & data[k];

        d = calculateEuclideanDistance(m_ftype, p0->dataptr, p1->dataptr);
        p1->datatoken_dist = d;
        ctotal += d;

        p0 = p1;
    }

    p1 = & data[0];

    d = calculateEuclideanDistance(m_ftype, p0->dataptr, p1->dataptr);
    p1->datatoken_dist = d;
    ctotal += d;

    return ctotal;
}

void CTSPHeuristicaOpenMP::resetResultData(datatokenptr_t* datares, datatokenptr_t* datawrk, long datasz)
{
    for(long i = 0; i < datasz; i++) {
        datatokenptr_t* p_wrk = & datawrk[i];
        p_wrk->dataptr->datatoken_processed = FALSE;
        p_wrk->dataptr->datatoken_order = i;
        p_wrk->dataptr->datatoken_dist = 0.0;
        p_wrk->datatoken_order = i;
        p_wrk->datatoken_dist = 0.0;

        datatokenptr_t* p_dst = & datares[i];
        p_dst->dataptr = p_wrk->dataptr;
        p_dst->datatoken_order = i;
        p_dst->datatoken_dist = 0.0;
    }
}

void CTSPHeuristicaOpenMP::execute()
{
    datatokenptr_t* p0;
    datatokenptr_t* p1;

    int nThreads = DEF_NUM_OF_THREADS;
    int tid;

    long tidDataSz[DEF_NUM_OF_THREADS];
    long tidStartData[DEF_NUM_OF_THREADS];

    long pos0, pos1;
    long i, k;

    double ctotal;

    resetResultData(m_datares, m_datawrk, m_datasz);

    for(k = 0; k < nThreads; k++)
    {
        tidDataSz[k] = (m_datasz / nThreads);
        tidStartData[k] = k * tidDataSz[k];

        if(k == (nThreads - 1))
            tidDataSz[k] = m_datasz - tidStartData[k];
    }

#pragma omp parallel num_threads(DEF_NUM_OF_THREADS)    \
            private(p0,                                 \
                    p1,                                 \
                    tid,                                \
                    pos0,                               \
                    pos1,                               \
                    i,                                  \
                    nThreads)                           \
            shared( m_datares,                          \
                    m_datawrk,                          \
                    m_datasz,                           \
                    tidDataSz,                          \
                    tidStartData)
    {
        //nThreads = omp_get_num_threads();
        tid = omp_get_thread_num();

        pos0 = tidStartData[tid];

        p0 = & m_datares[pos0];
        p0->datatoken_dist = 0.0;
        for(i = 1; i < tidDataSz[tid]; i++) 
        {
            pos1 = tidStartData[tid] + i;

            p1 = & m_datares[pos1];
            p1->datatoken_dist = 0.0;

            calculateMinCost(p0, p1, m_datawrk, m_datasz, tidStartData);
            p0 = p1;
        }
    
    }

    ctotal = calculateTotalCost(m_datawrk, m_datasz);

    m_result->datatest_val.datavaltest_initial_cost = ctotal; 
    m_result->datatest_val.datavaltest_final_cost = ctotal; 
}

void CTSPHeuristicaOpenMP::debugWorkData(datatokenptr_t* data, long size)
{
    printf("\n\nDEBUG: [ ");

    for(int m = 0; m < size; m++) {
        datatoken_t* pTmp = data[m].dataptr;
        printf("%ld; ", pTmp->datatoken_order);
    }

    printf(" ] ");
}
