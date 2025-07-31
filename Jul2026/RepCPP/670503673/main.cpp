#include<iostream>
#include<map>
#include<fstream>
#include<vector>
#include<utility>
#include<set>
#include<unordered_set>
#include<algorithm>
#include<random>
#include<numeric>
#include<getopt.h>
#include<mpi.h>
#include<omp.h>
#include<cmath>
#include<queue>
#include <unordered_map>


class Graph{
    public:
        int n, m ; // number of nodes and edges
        std::vector<std::unordered_set<int>> adj_list ; 
        std::vector<std::unordered_set<int>> adj_list_future ; 
        std::vector<int> degree ; // degree for each vertex, decrement if edge is deleted
        std::vector<bool> my_node ;
        std::vector<bool> my_neighbours ;
        std::vector<int> prio ; //priority of the node to determine ordering
        std::map<std::pair<int,int>, int> support_cnt ;
        std::vector<int> dsu_parent ;
        std::vector<int> dsu_rank ;
        std::vector<std::vector<int>> grp_rep_to_members;
        std::vector<omp_lock_t> v_lock ;
    Graph(){
        n = 0 ;
        m = 0 ;
    };

    Graph(int n, int m){
        this -> n = n ;
        this -> m = m ;
        this -> adj_list = std::vector<std::unordered_set<int>>(n); // adjaceny list assuming 0 ... n-1 numbering
        this -> adj_list_future = std::vector<std::unordered_set<int>>(n);
        this -> degree = std::vector<int>(n,0) ;
        this -> my_node = std::vector<bool>(n,false) ;
        this -> my_neighbours = std::vector<bool>(n,false) ;
        this -> prio = std::vector<int>(n) ;
        this -> dsu_parent = std::vector<int>(n,-1) ;
        this -> dsu_rank = std::vector<int>(n, 1) ;
        this -> grp_rep_to_members = std::vector<std::vector<int>>(n); 
        this -> v_lock = std::vector<omp_lock_t>(n);
        for(int i = 0 ; i < n ; i ++ )
        {
            omp_init_lock(&v_lock[i]) ;
        }
    } ; 
    

    inline bool order(int x, int y){
        // is x < y
        return prio[x] < prio[y] || (prio[x] == prio[y] and x < y);
    }

    inline void add_edge( int i, int j, bool myNode = true)
    {
        // i --> j edge added
        adj_list[i].insert(j) ;
        degree[i] ++ ; // degree of j will be incremented when j --> i is added
        if(myNode) my_node[i] = true ;
    }
   
    void prefilterp(int k, int& rank, int& num_procs)
    {
        // maintains my graph and my neighbour graph
        std::set<int> to_be_deleted ;
        std::set<int> send_to_master_set ;
        int nodes_with_larger_seg_size = n % num_procs ;
        int smaller_seg_size = (n/num_procs) ;
        int larger_seg_size = smaller_seg_size+1;        
        int start_i = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );
        int end_i = start_i + (rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size);
        while(true)
        {       
            // iterate over my nodes
            for( int node = start_i ; node < end_i ; node ++ )
            {
                if(degree[node] < k-1 and degree[node] > 0 ) {
                    to_be_deleted.insert(node) ;
                    send_to_master_set.insert(node) ;
                }
            }


            while(to_be_deleted.size()>0)
            {
                // delete this node, dequeue from to_be_deleted list
                int node = *to_be_deleted.begin();
                to_be_deleted.erase(to_be_deleted.begin());

                for(auto nbr : adj_list[node])
                {
                    // update adj list of neighbor
                    if(my_node[nbr])
                    {
                        if(adj_list[nbr].find(node) != adj_list[nbr].end()){
                            adj_list[nbr].erase(node);
                            degree[nbr]--;
                        }
                        if(degree[nbr] < k-1) {
                            to_be_deleted.insert(nbr) ; // nbr not already deleted
                            send_to_master_set.insert(nbr) ;
                        }
                    }
                }
                //remove node from my graph
                adj_list[node].clear();
                degree[node] = 0;
            }
            // share to master
            std::vector<int> recv_deleted_nodes, recv_counts(num_procs), displs(num_procs);
            
            int my_size = send_to_master_set.size();
            MPI_Allgather(&my_size, 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
            int tot_recvs = 0;
            for (int i = 0; i < num_procs; i++) {
                displs[i] = tot_recvs;
                tot_recvs += recv_counts[i];
            }
            recv_deleted_nodes.resize(tot_recvs);

            //exit condition
            if(tot_recvs == 0) break;
            
            std::vector<int> send_to_master(send_to_master_set.begin(), send_to_master_set.end());
            MPI_Allgatherv(&send_to_master[0], my_size, MPI_INT, &recv_deleted_nodes[0], &recv_counts[0], &displs[0], MPI_INT, MPI_COMM_WORLD);


            for(auto deleted_node : recv_deleted_nodes){
                if(my_neighbours[deleted_node]){
                    for(auto ele : adj_list[deleted_node]){
                        if(my_node[ele]){
                            adj_list[ele].erase(deleted_node);
                            degree[ele] = adj_list[ele].size();
                        }
                    }
                }
            }
            send_to_master.clear() ;
            send_to_master_set.clear() ;
            to_be_deleted.clear() ;
        }
    }

    inline void find_common_neighbours(int x, int y, std::vector<int>&nbrs)
    {
        if(adj_list[x].size() > adj_list[y].size()) std::swap(x,y) ;

        for(auto &ele  : adj_list[x])
        {
            if(adj_list[y].count(ele)) nbrs.push_back(ele) ;
        }

        return  ;
    }

    void ktrussp(int k, int& rank, int& num_procs)
    {
        // maintaints my graph and my neighbour graphs
        std::vector<int> send_to_master_array;
        std::vector<std::pair<int, int>> deleted_edges;

        int nodes_with_larger_seg_size = n % num_procs ;
        int smaller_seg_size = (n/num_procs) ;
        int larger_seg_size = smaller_seg_size+1;        
        int start_i = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );
        int end_i = start_i + (rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size);

        // INitialize support count for my edges
        bool first_time = support_cnt.empty() ; // haven't computed support count of my edges
        #pragma omp parallel for
        for(int node = start_i ; node < end_i ; node++)
        {
            // std::cout << omp_get_thread_num() << "\n" ; 
            for(auto nbr : adj_list[node]) 
            {
                if(order(node, nbr))
                {
                    // my edge, compute and store support count
                    if(first_time)
                    {
                        std::vector<int> nbrs ; find_common_neighbours(node,nbr,nbrs) ;
                        int common_neighbours_cnt =  nbrs.size() ;
                        #pragma omp critical
                        support_cnt[{node,nbr}] = common_neighbours_cnt ; // node < nbr
                    }
                    
                    #pragma omp critical
                    if(support_cnt[{node,nbr}] < k-2 ) // if i iterate over an edge it will not be deleted no need to check for deleted
                    {
                        send_to_master_array.push_back(node) ; send_to_master_array.push_back(nbr) ;
                    }
                }
            }
        }

        std::vector<int> recv_counts(num_procs);
        std::vector<int> displs(num_procs);
        bool flag = true;
        
        #pragma omp parallel shared(flag) 
        {
            flag = true ;
            int tid = omp_get_thread_num();
            while(flag)
            {
                #pragma omp barrier
                // communicated edges
                if(tid == 0 )
                {
                    int my_size = send_to_master_array.size();
                    MPI_Allgather(&my_size, 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
                    int tot_recvs = 0;
                    for (int i = 0; i < num_procs; i++) {
                        displs[i] = tot_recvs;
                        tot_recvs += recv_counts[i];
                    }
                    if(tot_recvs == 0){
                        flag = false ;
                    }

                    std::vector<int> recv_deleted_edges(tot_recvs);
                    MPI_Allgatherv(&send_to_master_array[0], my_size, MPI_INT, &recv_deleted_edges[0], &recv_counts[0], &displs[0], MPI_INT, MPI_COMM_WORLD);
                    deleted_edges.clear();
                    // #pragma omp parallel for
                    for(int i = 0; i < tot_recvs; i+=2){  // use displs
                            int x = recv_deleted_edges[i], y = recv_deleted_edges[i+1];
                            if(!my_node[x] and !my_node[y] and !my_neighbours[x] and !my_neighbours[y]) continue;  
                            
                            // #pragma omp critical
                            deleted_edges.push_back({x, y});
                            
                    }

                    send_to_master_array.clear() ;
                    recv_deleted_edges.clear() ;
                }

                #pragma omp barrier

                #pragma omp for
                for(int i = 0 ; i < deleted_edges.size() ; i++)
                {

                    int x = deleted_edges[i].first, y = deleted_edges[i].second;

                    #pragma omp critical
                    {
                        omp_set_lock(&v_lock[x]);  
                        omp_set_lock(&v_lock[y]);  
                    }

                    degree[x] -- ; degree[y] -- ;
                    adj_list[x].erase(y) ; adj_list[y].erase(x) ;

                    //update support counts
                    std::vector<int> support_nbrs; find_common_neighbours(x, y, support_nbrs);

                    omp_unset_lock(&v_lock[x]);
                    omp_unset_lock(&v_lock[y]);

                    for(auto support_nbr : support_nbrs){
                        int max_node, min_node ;
                        // process the edge (x, support_nbr);
                        if(order(x,support_nbr)){
                            max_node = support_nbr ;
                            min_node = x ;
                        }else{
                            max_node = x;
                            min_node = support_nbr ;
                        }

                        if(my_node[min_node])
                        {
                            // my_edge
                            #pragma omp critical
                            {
                                int supp_cnt = --support_cnt[{min_node, max_node}]  ;
                                // delete edge if already not deleted
                                if( supp_cnt == k-3 )
                                {
                                    send_to_master_array.push_back(min_node) ; send_to_master_array.push_back(max_node);
                                }
                            }
                        }
                        // process the edge (y,support_nbr) ;
                        if(order(y,support_nbr)){
                            max_node = support_nbr ;
                            min_node = y ;
                        }else{
                            max_node = y;
                            min_node = support_nbr ;
                        }

                        if(my_node[min_node])
                        {
                            // my_edge
                            #pragma omp critical
                            {
                                int supp_cnt = --support_cnt[{min_node, max_node}]  ;
                                // delete edge if already not deleted
                                if( supp_cnt == k-3 )
                                {
                                    send_to_master_array.push_back(min_node) ; send_to_master_array.push_back(max_node);
                                }
                            }
                        }
                    }
                        
                    
                }
            }
        }
    }
    
    inline bool any_edge_exists()
    {
        for( int i = 0 ; i < n ; i ++ )
        {
            if(my_node[i] and adj_list[i].size()>0) return true ;
        }
        return false ;
    }

    inline void make_copy(){
        adj_list_future = adj_list;
    }

    int find_dsu_parent(int node, std::vector<int>&dsu_array)
    { 
        // node is root if its parent is -1
        if(dsu_array[node] == -1)
        {
            // root 
            return node ;
        }
        // path compression
        return dsu_array[node] =  find_dsu_parent(dsu_array[node], dsu_array) ;
    }

    void dsu_union(int i, int j, std::vector<int>&dsu_array)
    {
        int par_i = find_dsu_parent(i, dsu_array), par_j = find_dsu_parent(j, dsu_array) ;
        if(par_i == par_j)
        {
            return ; 
        }
        // union by rank
        if (dsu_rank[par_j] < dsu_rank[par_i]){
            std::swap(par_i, par_j);
        }
        dsu_array[par_i] = par_j ; 
        dsu_rank[par_j] += dsu_rank[par_i];

        return ;
    }

    void dsu(int rank, int num_procs)
    {
        this->dsu_parent = std::vector<int>(n,-1);
        int nodes_with_larger_seg_size = n % num_procs ;
        int smaller_seg_size = (n/num_procs) ;
        int larger_seg_size = smaller_seg_size+1;
        int my_seg_size = rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
        int seg_offset = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );
        for( int i = seg_offset ; i < seg_offset + my_seg_size ; i ++  )
        {
            // my nodes

            for(auto nbr : adj_list[i])
            {
                dsu_union(i, nbr, dsu_parent) ;
            }
        }

    }
    
    void synch_dsu(int rank, int num_procs)
    {
        // find dsu_parent array of my own graph
        dsu(rank, num_procs) ;

        if(num_procs == 1) return ;
        // 0 -> 1 .... -> n-1 
        if(rank == 0)
        {
            // only send
            MPI_Send(&dsu_parent[0], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }else if(rank < num_procs-1 ){
            // receive
            std::vector<int> nbr_dsu_parent(n);
            MPI_Recv(&nbr_dsu_parent[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ; 
            // join
            for( int i = 0 ; i < n ; i ++ )
            {
                if(nbr_dsu_parent[i] != -1)
                dsu_union(i, find_dsu_parent(i, nbr_dsu_parent), dsu_parent);
            }
            // send
            MPI_Send(&dsu_parent[0], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }else{
            // last node
            // receive
            std::vector<int> nbr_dsu_parent(n);
            MPI_Recv(&nbr_dsu_parent[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ; 
            // join
            for( int i = 0 ; i < n ; i ++ )
            {
                if(nbr_dsu_parent[i] != -1)
                dsu_union(i, find_dsu_parent(i, nbr_dsu_parent), dsu_parent);
            }
        }
    }

    void find_groups()
    {
        for(int i = 0 ; i < n ; i ++ )
        {
            int grp_rep = find_dsu_parent(i, dsu_parent);
            grp_rep_to_members[grp_rep].push_back(i) ;
        }
    }

    std::pair<int,int> connected_components(std::unordered_map<int, std::vector<int>> &cc_s)
    {
        
        for(int i = 0 ; i < n ; i ++ )
        {
            cc_s[find_dsu_parent(i, dsu_parent)].push_back(i) ;
        }

        int cnt=0, one_cnt = 0;

        for(auto pr : cc_s)
        {
            if(pr.second.size() > 1) cnt ++ ;
            else one_cnt ++ ;
        }

        return {cnt, one_cnt} ;
        

    }
   
    void task2(int& rank, int& num_procs, int& p, std::string& outputpath, int verbose){

        // give the dsu arrays for the entire graph from the last node to all nodes
        MPI_Bcast(&dsu_parent[0], n, MPI_INT, num_procs-1, MPI_COMM_WORLD);
        MPI_Bcast(&dsu_rank[0], n, MPI_INT, num_procs-1, MPI_COMM_WORLD);

        int nodes_with_larger_seg_size = n % num_procs ;
        int smaller_seg_size = (n/num_procs) ;
        int larger_seg_size = smaller_seg_size+1;
        int my_seg_size = rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
        int seg_offset = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );
        std::vector<std::unordered_set<int>> grp_reps(my_seg_size);

        // iterate on my nodes
        #pragma omp parallel for
        for( int i = seg_offset ; i < seg_offset + my_seg_size ; i ++){
            // use the original edges
            for(auto nbr : adj_list_future[i]){
                int pr = find_dsu_parent(nbr, dsu_parent);
                // if neighbour is not an isloted k-truss
                if(dsu_rank[pr] > 1){
                    grp_reps[i-seg_offset].insert(pr);
                }
            }
        }

        int cnt = 0, cnt_sum = 0;
        for(auto se : grp_reps){
            if(se.size() >= p) {
                cnt++ ;
            }
        }
        MPI_Reduce(&cnt, &cnt_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        std::ofstream fw;
        if(rank == 0){
            fw.open(outputpath, std::ios::app);
            fw << cnt_sum << "\n";
            fw.close();
        }

        
        // find group rep ==> group needed for verbose 0
        if(verbose == 1) find_groups();

        if(rank == 0){
            
            fw.open(outputpath, std::ios::app);
            //write
            int token = 1;
            for( int i = seg_offset ; i < seg_offset + my_seg_size ; i ++){
                if(grp_reps[i-seg_offset].size() >= p)
                {
                    if(verbose == 0) fw << i << " ";
                    else if(verbose==1)
                    {
                        fw << i << "\n" ;
                        // print all vertices of neighbouring groups
                        for(auto rep : grp_reps[i-seg_offset])
                        {
                            for(auto member : grp_rep_to_members[rep]) fw << member << " " ;
                        }
                        fw << "\n";
                    }
                } 
            }
            fw.close();

            if(num_procs == 1) return;

            MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        else if (rank < num_procs){
            int token = 1;
            MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            fw.open(outputpath, std::ios::app);

            for(int i = seg_offset ; i < seg_offset + my_seg_size ; i ++){
                if(grp_reps[i-seg_offset].size() >= p)
                {
                    if(verbose == 0) fw << i << " ";
                    else if(verbose==1)
                    {
                        fw << i << "\n" ;
                        // print all vertices of neighbouring groups
                        for(auto rep : grp_reps[i-seg_offset])
                        {
                            for(auto member : grp_rep_to_members[rep]) fw << member << " " ;
                        }
                        fw << "\n";
                    }
                } 
            }

            fw.close();

            if(rank != num_procs-1) MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
};


void read_filep(const char* inpfile, const char* offsetfile, Graph& g, int& rank, int& num_procs){
    // reads graph corresponding to my own nodes only
    MPI_File inps;
    MPI_File_open(MPI_COMM_WORLD, inpfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &inps);
    MPI_File offsets;
    MPI_File_open(MPI_COMM_WORLD, offsetfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &offsets);

    int ints[2]; // (n, m) i.e. num_vertices, num_edges
    if(rank == 0){
        //reading n, m
        MPI_File_read(inps, ints, 2, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(ints, 2, MPI_INT, 0, MPI_COMM_WORLD);

    g = Graph(ints[0], ints[1]);

    int nodes_with_larger_seg_size = ints[0] % num_procs ;
    int smaller_seg_size = (ints[0]/num_procs) ;
    int larger_seg_size = smaller_seg_size+1;
    int my_seg_size = rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
    int seg_offset = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );

    int* my_offsets = (int*) malloc(sizeof(int) * my_seg_size);
    MPI_File_read_at_all(offsets, seg_offset * sizeof(int), my_offsets, my_seg_size, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Offset offset;

    std::vector<int> my_degrees(my_seg_size);

    for(int i = 0; i < my_seg_size; i++){
        int off = my_offsets[i];
        MPI_File_seek(inps, off, MPI_SEEK_SET);
        MPI_File_read(inps, ints, 2, MPI_INT, MPI_STATUS_IGNORE); //inps now have i, deg(i)
        my_degrees[i] = ints[1]; //storing my degree;

        int* node_info = (int*) malloc(sizeof(int) * ints[1]);
        MPI_File_read(inps, node_info, ints[1], MPI_INT, MPI_STATUS_IGNORE);
        for(int j = 0; j < ints[1]; j++) {
            g.add_edge(ints[0], node_info[j], true); // my own node, degree of my node and its adjacency list will be updated
        }
    }

    std::vector<int> recv_counts(num_procs);
    std::vector<int> displs(num_procs);

    int tot_recvs = 0;
    for (int i = 0; i < num_procs; i++) {
        recv_counts[i] = (i < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size);
        displs[i] = tot_recvs;
        tot_recvs += recv_counts[i];
    }

    MPI_Allgatherv(&my_degrees[0], my_seg_size, MPI_INT, &g.prio[0], &recv_counts[0], &displs[0], MPI_INT, MPI_COMM_WORLD);
    
    MPI_File_close(&inps);
    MPI_File_close(&offsets);

    g.make_copy();
    return;
}

void read_cl_arguments(int argc, char *argv[], int &taskid, std::string &inputpath, std::string &headerpath, std::string &outputpath, int &verbose, int &startk, int &endk, int &p )
{

    static struct option long_options[] = {
        {"taskid", required_argument, 0, 't'},
        {"inputpath", required_argument, 0, 'i'},
        {"headerpath", required_argument, 0, 'h'},
        {"outputpath", required_argument, 0, 'o'},
        {"verbose", required_argument, 0, 'v'},
        {"startk", required_argument, 0, 's'},
        {"endk", required_argument, 0, 'e'},
        {"p", required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };

    // Parse the command-line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "t:i:h:o:v:s:e:p:", long_options, &option_index)) != -1) {
        switch (c) {
        case 't':
            taskid = atoi(optarg);
            break;
        case 'i':
            inputpath = optarg;
            break;
        case 'h':
            headerpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'v':
            verbose = atoi(optarg);
            break;
        case 's':
            startk = atoi(optarg);
            break;
        case 'e':
            endk = atoi(optarg);
            break;
        case 'p':
            p = atoi(optarg);
            break;
        case '?':
            std::cerr << "Invalid command-line argument" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void get_neighbour_graph(Graph &g, int rank, int num_procs)
{
    // information exchange between all processors
    // eventually each process will receive all edges of neighbours its own nodes
    int n = g.n ;
    int nodes_with_larger_seg_size = n % num_procs ;
    int smaller_seg_size = (n/num_procs) ;
    int larger_seg_size = smaller_seg_size+1;
    int my_seg_size = rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
    int seg_offset = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );

    std::vector<std::vector<int>> temp_adj_list(n);
    // ITERATE OVER MY NODES
    for(int i = seg_offset ; i < seg_offset + my_seg_size ; i ++ )
    {
        for(auto ele : g.adj_list[i])
        {
            temp_adj_list[i].push_back(ele) ;
            if(!g.my_node[ele]) { // NEIGHBOUR BUT NOT MY NODE
                g.my_neighbours[ele] = true;
            }
        }
    }
    int size_of_adj_list ;
    // iterate over all processes
    for( int rank_of_process = 0 ; rank_of_process < num_procs ; rank_of_process ++ )
    {
        // iterate over all nodes of a process
        int start_i = (rank_of_process < nodes_with_larger_seg_size ? rank_of_process*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank_of_process-nodes_with_larger_seg_size)*smaller_seg_size );
        int end_i = start_i + (rank_of_process < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size);

        for(int i = start_i ; i < end_i ; i ++ )
        {
            if(rank == rank_of_process){
                // my node (node of the process itself)
                size_of_adj_list = temp_adj_list[i].size() ;
            }
            MPI_Bcast(&size_of_adj_list, 1, MPI_INT, rank_of_process, MPI_COMM_WORLD) ;
            temp_adj_list[i].resize(size_of_adj_list) ;
            MPI_Bcast(&temp_adj_list[i][0], size_of_adj_list, MPI_INT, rank_of_process, MPI_COMM_WORLD) ;
            if(rank != rank_of_process and g.my_neighbours[i])
            {
                // not my node but my neighbour => add to graph
                g.adj_list[i].clear() ;
                g.degree[i] = 0 ;
                for(auto ele : temp_adj_list[i])
                {
                    g.add_edge(i, ele, false) ; // not my node
                }
            }
        }
    }
}

void send_graph_to_last_node(Graph &g, int rank, int num_procs)
{
    int n = g.n ;
    int nodes_with_larger_seg_size = n % num_procs ;
    int smaller_seg_size = (n/num_procs) ;
    int larger_seg_size = smaller_seg_size+1;
    int my_seg_size = rank < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
    int seg_offset = (rank < nodes_with_larger_seg_size ? rank*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank-nodes_with_larger_seg_size)*smaller_seg_size );
    if(rank != num_procs-1)
    {
        // send adj list
        // iterate over my nodes and send them
        for(int i = seg_offset ; i < seg_offset + my_seg_size ; i++)
        {
            std::vector<int>temp_adj_list ; 
            for(auto ele : g.adj_list[i])
            {
                temp_adj_list.push_back(ele) ;
            }
            int adj_list_size = temp_adj_list.size() ;
            MPI_Send(&adj_list_size, 1, MPI_INT, num_procs-1, 0, MPI_COMM_WORLD) ;
            MPI_Send(&temp_adj_list[0], adj_list_size, MPI_INT, num_procs-1, 0, MPI_COMM_WORLD) ;
        }
    }else{
        // recv adj list
        for(int rank_sender = 0 ; rank_sender < num_procs - 1 ; rank_sender ++ )
        {
            int sender_size =  rank_sender < nodes_with_larger_seg_size ? larger_seg_size : smaller_seg_size;
            int sender_offset =  (rank_sender < nodes_with_larger_seg_size ? rank_sender*larger_seg_size : (nodes_with_larger_seg_size*larger_seg_size)+ (rank_sender-nodes_with_larger_seg_size)*smaller_seg_size );
            for(int i = sender_offset ; i < sender_size + sender_offset ; i ++ )
            {
                // not my node receive their adj_list
                int adj_list_size;
                std::vector<int>temp_adj_list ;
                MPI_Status status ;
                MPI_Recv(&adj_list_size, 1, MPI_INT, rank_sender, 0, MPI_COMM_WORLD, &status) ;
                temp_adj_list.resize(adj_list_size) ;
                MPI_Recv(&temp_adj_list[0], adj_list_size, MPI_INT, rank_sender, 0, MPI_COMM_WORLD, &status) ;

                g.adj_list[i].clear() ;
                g.degree[i] = 0 ;
                for(auto ele : temp_adj_list)
                {
                    g.add_edge(i, ele, false) ; // not my node
                }
            }
        }
    }
}


bool gather_edge_exists_at_last_node(Graph &g, int rank, int num_procs)
{
    int edge_exists = g.any_edge_exists() ;
    bool answer = edge_exists ;
    if(num_procs == 1) return answer ;
    int* recv_buff = NULL ;
    if(rank == num_procs-1) recv_buff = new int[num_procs] ;

    MPI_Gather(&edge_exists, 1, MPI_INT, recv_buff, 1, MPI_INT, num_procs-1, MPI_COMM_WORLD) ;
    
    if(rank == num_procs-1)
    {
        for(int i = 0 ; i < num_procs ; i ++ )
        {
            if(recv_buff[i]) answer = true ;
            if(answer) break ;
        }
        delete[] recv_buff ;
    }

    return answer ;
}

int main(int argc, char *argv[])
{
    
    int rank, size;
    MPI_Status status;
    
    // Initialize MPI
    int provided , required = MPI_THREAD_MULTIPLE ;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
        printf("MPI_Init_thread failed to provide the required thread support level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // set number of threads for openMP

    int taskid = -1, verbose=-1, startk=-1, endk=-1, p=-1 ; 
    int num_threads = size/2 ;
    if(num_threads <= 0 ) num_threads = 1 ;
    std::string inputpath, headerpath, outputpath;
    read_cl_arguments(argc, argv, taskid, inputpath, headerpath, outputpath, verbose, startk, endk, p) ;
    omp_set_num_threads(num_threads);

    Graph g ;
    // read only my graph
    read_filep(inputpath.c_str(), headerpath.c_str(), g, rank, size);

    get_neighbour_graph(g, rank, size) ;

    std::ofstream fw(outputpath) ;

    bool zero_found = false,  edge_exists = false;

    if(taskid == 1){
        for(int i = startk+2 ; i <= endk+2 ; i ++ )
        {
            // MAINTAINS MY AND MY NEIGHBOURS GRAPH AND DELETES ALL POSSIBLE EDGES
            // only do this if zero not found i.e. graph still connected
            if(not zero_found){
                g.prefilterp(startk + 2, rank, size) ;
                g.ktrussp(i, rank, size);

                // if(!verbose) edge_exists = gather_edge_exists_at_last_node(g, rank, size) ; // less effort for non verbose
                // else send_graph_to_last_node(g, rank, size) ;

                if(!verbose) edge_exists = gather_edge_exists_at_last_node(g, rank, size) ; // less effort for non verbose
                else g.synch_dsu(rank, size) ;

            }

            
            // only one process prints zero
            if(zero_found)
            {
                if(rank == size-1 && verbose) fw << 0 << "\n" ;
                if(rank == size-1 && !verbose) fw << 0 << " " ;
                continue ;
            }

            // zero not found 1 process checks connected components
            std::unordered_map<int, std::vector<int>> connected_components ; // stores all connected components
            std::pair<int,int> cc_cnt = std::make_pair(0,0);
            if(rank == size-1)
            {
                if(!verbose)
                {
                    cc_cnt.first = edge_exists ; // only check if edge exists
                }else{
                    cc_cnt = g.connected_components(connected_components) ; // have to find cc
                }
                zero_found = (cc_cnt.first==0 ? true : false ) ;
            }
            // broadcast whether zero found or not
            MPI_Bcast(&zero_found, 1, MPI_BYTE, size-1, MPI_COMM_WORLD) ;
            if(rank != size-1) continue ; // last node prints
            if(verbose) fw << (cc_cnt.first ? 1 : 0 )  << "\n" ;
            if(!verbose) fw << (cc_cnt.first ? 1 : 0 )  << " " ;
            if(verbose)
            {
                if(cc_cnt.first ) fw << cc_cnt.first << "\n" ;
                for(auto &cc : connected_components)
                {
                    sort(cc.second.begin(), cc.second.end()) ;
                    if(cc.second.size()>1)
                    {
                        for(int i = 0 ; i < (int)cc.second.size() ; i ++ )
                        {
                            fw << cc.second[i] ;
                            if(i != (int)cc.second.size()-1) fw << " " ;
                            else fw << "\n" ;
                        }
                    }
                }
            }
        }
        
        if(!verbose && rank==size-1) fw << "\n" ;
        fw.close() ;
    }

    if(taskid == 2){
        // everybody now has the dsu array of the entire graph
        g.prefilterp(endk + 2, rank, size) ;
        g.ktrussp(endk + 2, rank, size);
        g.synch_dsu(rank, size);
        g.task2(rank, size, p, outputpath, verbose);
    }

    MPI_Finalize();
    return 0 ;
}