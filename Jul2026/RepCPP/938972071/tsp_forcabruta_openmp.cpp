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
 * tsp_forcabruta_openmp.cpp
 */

#include"tsp_all.h"

CTSPForcaBrutaOpenMP::CTSPForcaBrutaOpenMP(
    int appid, 
    int nprocs, 
    int ftype, 
    datatest_t* result, 
    datatokenptr_t** datawrk, 
    datatokenptr_t** datares, 
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

CTSPForcaBrutaOpenMP::~CTSPForcaBrutaOpenMP()
{
    //TODO:
}

/* Methodes */

double CTSPForcaBrutaOpenMP::calculateCost(datatokenptr_t* datawrk, long datasz)
{
    double dist_total = 0.0;
    double d = 0.0;

    datatokenptr_t* pI = & datawrk[0];

    datatokenptr_t* p0 = pI;
    p0->datatoken_dist = 0;
    for(long i = 1; i < datasz; i++) {
        datatokenptr_t* p1 = & datawrk[i];

        d = calculateEuclideanDistance(m_ftype, p0->dataptr, p1->dataptr);
        dist_total += d;

        p1->datatoken_dist = d;

        p0 = p1;
    }

    d = calculateEuclideanDistance(m_ftype, p0->dataptr, pI->dataptr);
    dist_total += d;

    pI->datatoken_dist = d;

    return dist_total;
}

void CTSPForcaBrutaOpenMP::avaliateTotalCost(int tid, double* initial_cost, double* min_cost, datatokenptr_t* datares, datatokenptr_t* datawrk, long datasz)
{
    double d = calculateCost(datawrk, datasz);

    //debugWorkData(tid, datawrk, datasz);
    //getchar();

    if(d < (*min_cost)) {
        (*min_cost) = d;
        copyWorkToResultData(datares, datawrk, datasz);
    
        if((*initial_cost) < 0.05)
            (*initial_cost) = d;
    }
}

void CTSPForcaBrutaOpenMP::copyWorkToResultData(datatokenptr_t* datares, datatokenptr_t* datawrk, long datasz)
{
    for(long i = 0; i < datasz; i++) {
        datatokenptr_t* p_wrk = & datawrk[i];        
        datatokenptr_t* p_dst = & datares[i];
        
        p_dst->dataptr = p_wrk->dataptr;
        p_dst->datatoken_order = p_wrk->datatoken_order;
        p_dst->datatoken_dist = p_wrk->datatoken_dist;
    }
}

void CTSPForcaBrutaOpenMP::mergeResultData(datatokenptr_t* p_dst, datatokenptr_t* p_src, long datasz)
{    
    for(long i = 0; i < datasz; i++) {
        p_dst[i].dataptr = p_src[i].dataptr;
        p_dst[i].datatoken_order = p_src[i].datatoken_order;
        p_dst[i].datatoken_dist = p_src[i].datatoken_dist;
    }
}

void CTSPForcaBrutaOpenMP::resetResultData(datatokenptr_t* datares, datatokenptr_t* datawrk, long datasz)
{
    for(long i = 0; i < datasz; i++) {
        datatokenptr_t* p_wrk = & datawrk[i];
        p_wrk->datatoken_initial_order = i;
        p_wrk->datatoken_order = i;
        p_wrk->datatoken_dist = 0.0;

        p_wrk->dataptr = p_wrk->initial_dataptr;        
        p_wrk->dataptr->datatoken_processed = FALSE;
        p_wrk->dataptr->datatoken_order = i;
        p_wrk->dataptr->datatoken_dist = 0.0;

        datatokenptr_t* p_dst = & datares[i];
        p_dst->datatoken_initial_order = i;
        p_dst->datatoken_order = i;
        p_dst->datatoken_dist = 0.0;

        p_dst->initial_dataptr = p_wrk->initial_dataptr;
        p_dst->dataptr = p_dst->initial_dataptr;
    }
}

long CTSPForcaBrutaOpenMP::validData(datatokenptr_t* datawrk, long datasz)
{
    for(long i = 0; i < datasz - 1; i++) {
        datatokenptr_t* p1 = & datawrk[i];
        long pos2 = p1->datatoken_order;

        datatokenptr_t* p2 = & datawrk[pos2];
        p1->dataptr = p2->initial_dataptr;

        for(long j = i + 1; j < datasz; j++) {
            datatokenptr_t* p_tmp = & datawrk[j];
            if(p1->datatoken_order == p_tmp->datatoken_order)
                return FALSE;
        }
    }

    datatokenptr_t* p1 = & datawrk[datasz - 1];
    long pos2 = p1->datatoken_order;

    datatokenptr_t* p2 = & datawrk[pos2];
    p1->dataptr = p2->initial_dataptr;

    return TRUE;
}

long CTSPForcaBrutaOpenMP::executeStep(long inival, long finval, datatokenptr_t* datawrk, long datasz)
{
    long isRunning = TRUE;

    long max_val = datasz - 1;
    long last_pos = datasz - 1;

    for(long i = 1; i < datasz; i++)
    {
        datatokenptr_t* p1 = & datawrk[i];

        long initial_order1 = p1->datatoken_initial_order;

        long new_step1 = p1->datatoken_step + 1;
        if((i == 1) && (new_step1 < inival)) 
            new_step1 = inival;

        if(new_step1 <= finval) {
            long new_order1 = (initial_order1 + new_step1) % (max_val + 1);

            p1->datatoken_step = new_step1;
            p1->datatoken_order = new_order1;

            break;
        }
        else {
            new_step1 = (i == 1) ? inival : 0;

            long new_order1 = (initial_order1 + new_step1) % (max_val + 1);

            p1->datatoken_step = new_step1;
            p1->datatoken_order = new_order1;

            if(i == last_pos) {
                isRunning = FALSE;
                break;
            }
        }
    }

    return isRunning;
}

void CTSPForcaBrutaOpenMP::execute()
{
    datatokenptr_t* p_datares;
    datatokenptr_t* p_datawrk;
    long datasz;

    double initial_cost[DEF_NUM_OF_THREADS];
    double final_cost[DEF_NUM_OF_THREADS];

    double inicost;
    double fincost;

    int nThreads;
    int tid;

    long tidLenVal;
    long tidStartVal;

    long inival;
    long finval;

    long isRunning;

    long bIsValid;

#pragma omp parallel num_threads(DEF_NUM_OF_THREADS)    \
            private(p_datares,                          \
                    p_datawrk,                          \
                    datasz,                             \
                    tid,                                \
                    tidLenVal,                          \
                    tidStartVal,                        \
                    inival,                             \
                    finval,                             \
                    inicost,                            \
                    fincost,                            \
                    isRunning,                          \
                    bIsValid)                           \
            shared( m_datares,                          \
                    m_datawrk,                          \
                    m_datasz,                           \
                    initial_cost,                       \
                    final_cost,                         \
                    nThreads)
    {
        nThreads = omp_get_num_threads();
        tid = omp_get_thread_num();

        tidLenVal = ((m_datasz - 1) / nThreads);
        tidStartVal = tid * tidLenVal;

        if(tid == (nThreads - 1))
            tidLenVal = (m_datasz - 1) - tidStartVal;

        inival = tidStartVal;
        finval = (tidStartVal + tidLenVal);

        initial_cost[tid] = 0.0;
        final_cost[tid] = 0.0;

        p_datares = m_datares[tid];
        p_datawrk = m_datawrk[tid];
        datasz = m_datasz;

        resetResultData(p_datares, p_datawrk, datasz);

        inicost = 0.0;
        fincost = DEF_MAX_VALUE;

        isRunning = TRUE;
        while(isRunning == TRUE) 
        {
            bIsValid = validData(p_datawrk, m_datasz);
            if(bIsValid == TRUE) {
                avaliateTotalCost(tid, & inicost, & fincost, p_datares, p_datawrk, datasz);
            }
            isRunning = executeStep(inival, finval, p_datawrk, datasz);
        }

        initial_cost[tid] = inicost;
        final_cost[tid] = fincost;
    }

    datatokenptr_t* p_dst = m_datares[0];

    datavaltest_t* p_tst = & m_result->datatest_val;
    p_tst->datavaltest_initial_cost = initial_cost[0];
    p_tst->datavaltest_final_cost = DEF_MAX_VALUE;

    for(int i = 0; i < DEF_NUM_OF_THREADS; i++) 
    {
        if(final_cost[i] < p_tst->datavaltest_final_cost) 
        {
            p_tst->datavaltest_final_cost = final_cost[i];
    
            datatokenptr_t* p_src = m_datares[i];
            mergeResultData(p_dst, p_src, m_datasz);
        }
    }
}

void CTSPForcaBrutaOpenMP::debugWorkData(int tid, datatokenptr_t* datawrk, long datasz)
{
    double ctotal = 0.0;

    printf("\n\nDEBUG[TID=%d]: [ ", tid);

    long size = MIN(datasz, DEF_SHOWDATA_MAX_ROWS);

    for(long i = 0; i < size; i++) {
        datatokenptr_t* p = & datawrk[i];
        printf("%ld;", p->datatoken_order);
        ctotal += p->datatoken_dist;        
    }

    printf(" ..... ");

    for(long i = 0; i < size; i++) {
        datatokenptr_t* p = & datawrk[i];
        printf("%ld;", p->datatoken_step);
        ctotal += p->datatoken_dist;        
    }

    printf(" ..... COST: %lf ] ", ctotal);

}
