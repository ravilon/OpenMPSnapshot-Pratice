#ifdef _OPENMP
# include <omp.h>
#endif

#include <iostream>

using std::cout;
using std::endl;
using std::to_string;
using std::string;

#include <fstream>
#include <utility>
#include <set>
#include <vector>  //vectors
using std::vector;

#include <iomanip> //set precision
#include <algorithm>
#include <random> // probability distribution
#include <unistd.h>

#include "Agents.h"
int na = pow(2, 20);
//const int nbgha = 100;
const int agentcounta = 100;
int matrixa[100][100]={};

void open_space_scores(int j, vector<double> &input_vec,int k) {
    std::ifstream in_scores;
    double i_score;
    in_scores.open("NK_space_scores_"+to_string(k)+"_"+to_string(j)+".txt",std::ios::in | std::ios::binary);
    //while (in_scores>>i_score) {input_vec[i] = i_score; ++i; cout<<input_vec[i]<<endl;}
    for (int i = 0; i < ::na; ++i) {
        in_scores >> i_score;
        //cout<<i_score<<"test"<<'\n';
        input_vec[i] = i_score;
        //cout<<input_vec[i]<<'\n';
    }
    in_scores.close();
}

void open_space_string(vector <string> &string_vec) {
    std::fstream strings;
    strings.open("NK_space_strings.txt");
    for (int i = 0; i < ::na; ++i) {
        string i_str;
        strings >> i_str;
        string_vec[i] = i_str;
    }
    strings.close();
}

void A_list(vector<char> &types){//creates list of A and B and  assigns one to an agent at the beginning of every new NK_Space
    std::vector<char> sp(100);
    std::fill(sp.begin(),sp.end(),'A');
    for (unsigned int i = 0; i < sp.size(); ++i)
    {
        types[i]=sp[i];
    }   
}
void AB_list(vector<char> &types){//creates list of A and B and  assigns one to an agent at the beginning of every new NK_Space
    std::vector<char> sp={'A','A','B','B','B','A','A','B','A','B','B','A','A','A','B','B','B','B','A','A','B','B','A','B','B','A','A','B','B','B','B','B','B','B','A','A','A','B','B','B','A','A','A','B','B','B','A','A','B','B','B','B','A','A','B','B','A','B','B','A','B','A','B','A','A','B','B','A','B','A','B','A','A','B','A','B','B','A','A','B','B','A','B','B','A','B','A','B','B','B','B','A','A','A','A','B','B','A','A','B'};

    for (unsigned int i = 0; i < sp.size(); ++i)
    {
        types[i]=sp[i];
    }   
}

void AB_part_25_list(vector<char> &types){//creates list of A and B and  assigns one to an agent at the beginning of every new NK_Space
    std::vector<char> sp={'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'};

    for (unsigned int i = 0; i < sp.size(); ++i)
    {
        types[i]=sp[i];
    }   
}

void AB_part_5_list(vector<char> &types){//creates list of A and B and  assigns one to an agent at the beginning of every new NK_Space
    std::vector<char> sp={'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'};
    for (unsigned int i = 0; i < sp.size(); ++i)
    {
        types[i]=sp[i];
    }   
}

void AB_random_list(vector<char> &types, int inksp){//Random assignment of A and B
	    int ac=0;
    	int bc=0;
    	int temp=0;
    	for (int i = 0; i < 100; ++i)
    	{
    	std::bernoulli_distribution d(0.5);
    	std::mt19937 gen;
    	gen.seed(inksp+i);
    	
    	temp=d(gen);
    	if(temp==1) {types[i]='A'; ac++;}
    	if(temp==0) {types[i]='B'; bc++;}
    	//cout<<temp<< ":V C:"<<types[i]<<" ";
    	d.reset();
    	}
    	//cout<<endl;
    	//cout<<ac<<":A B:"<<bc<<endl;
    	//std::cin.get();
}


void ABCDE_list(vector<char> &types){//creates list of A, B,C,D,E and randomly assigns one to an agent at the beginning of every new NK_Space
    vector<char> AB(::agentcounta);
    for (int i = 0; i < ::agentcounta; i++) {
        if (i%5==0) AB[i] = 'A';
        if (i%5==1) AB[i] = 'B';
        if (i%5==2) AB[i] = 'C';
        if (i%5==3) AB[i] = 'D';
        if (i%5==4) AB[i] = 'E';
    }
    int acount=0;
    int bcount=0;
    int ccount=0;
    int dcount=0;
    int ecount=0;
    int temp = 0;
    for (int i = 0; i < ::agentcounta; ++i) {
        temp = rand() % (::agentcounta - i);
        types[i] = AB[temp];
        AB.erase(AB.begin() + temp);
        if(types[i]=='A') acount++;
        //cout<<types[i]<<" ";
        if(types[i]=='B') bcount++;
        if(types[i]=='C') ccount++;
        if(types[i]=='D') dcount++;
        if(types[i]=='E') ecount++;
        
    }
}
void AB_list_deseg(vector<char> &types){//creates list of A and B and randomly assigns one to an agent at the beginning of every new NK_Space
    vector<char> AB(::agentcounta);
    for (int i = 0; i < ::agentcounta; ++i) {
        if (i%2==0) types[i] = 'A';
        if (i%2==1) types[i] = 'B';
    }
}

void AB_list_seg(vector<char> &types){//creates list of A and B and randomly assigns one to an agent at the beginning of every new NK_Space
    //vector<char> AB(::agentcounta);
    for (int i = 0; i < ::agentcounta; ++i) {
        if (i < 50) types[i] = 'A';
        if (i >= 50) types[i] = 'B';
    }
}

void AB_list_permute(vector<char> &types,int inksp){
	std::mt19937 gen;
	gen.seed(inksp+1);
	for (int i = 0; i < ::agentcounta; ++i) {
        if (i < 50) types[i] = 'A';
        if (i >= 50) types[i] = 'B';
    }
	std::shuffle ( types.begin(), types.end() ,gen);
}

void agent_connections_replacement_number(vector<Agent> &Agents,int inksp){
	std::mt19937 gen;
	gen.seed(inksp+1);
	std::vector<int> A_number={1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,8};
	std::vector<int> B_number={1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,8};
	std::shuffle ( A_number.begin(), A_number.end(),gen);
	std::shuffle ( B_number.begin(), B_number.end(),gen);
	int counter=0;
	for (int i = 0; i < 100; ++i){
		
		if(Agents[i].species=='A'){
			Agents[i].connection_replace=A_number[counter];
			counter++;
		}
		//cout<<counter<<',';
	
	}
	cout<<endl;
	counter=0;	
	for (int i = 0; i < 100; ++i){
		
		if(Agents[i].species=='B'){
			Agents[i].connection_replace=B_number[counter];
			counter++;
		}
		//cout<<counter<<',';
	}
}

void output_matrix_connections(int NKspace,vector<Agent> Agents,int rounds)
    {// using for checking if agent_swap_hack and agent_swap_con work correctly 
        std::ofstream out;
        std::ostringstream str;
        str << std::setw(3) << std::setfill('0') << NKspace;
        out.open("matrix_connections_"+str.str()+"_"+to_string(rounds)+".txt");
        for (int i = 0; i < 100; ++i)
        {
            if(i==0)out<<",";
            out<<"cted "<<i;
            if(i<=98) out<<",";
            if(i==99) out<<"\n";
        }
        for (int i = 0; i < 100; ++i)
        {
           out<<i<<",";
           for (int j = 0; j < 100; ++j)
           {
              
              for (int k = 0; k < 8; ++k)
              {
                  if(Agents[i].connections[k]==j) out<<j;
                  
              }
              out<<",";
              if(j==99)out<<"\n";
           }
        }
    }
void output_connections(int NKspace,vector<Agent> Agents,int rounds){ //outputs all agents connections as a csv file; includes minority status
	std::ofstream out;
	out.open("agent connections "+to_string(NKspace)+" "+to_string(rounds)+".txt");
	out<<"Agent id #"<<","<<"species"<<","<<"connection 0"<<","<<"connection 1"<<","<<"connection 2"<<","<<"connection 3"<<","<<"connection 4"<<","<<"connection 5"<<","<<"connection 6"<<","<<"connection 7"<<","<<"minority status"<<"\n";
	for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
	{
		out<<i->id<<","<<i->species<<","<<i->connections[0]<<","<<i->connections[1]<<","<<i->connections[2]<<","<<i->connections[3]<<","<<i->connections[4]<<","<<i->connections[5]<<","<<i->connections[6]<<","<<i->connections[7];
		if(i->minority==1) 
			{out<<","<<"M"<<"\n";} 
		else
			{out<<"\n";}
	}
}

void output_scores(int NKspace,vector<Agent> Agents,int rounds){ //outputs all agents connections as a csv file; includes minority status
	std::ofstream out;
	out.open("agent scores round "+to_string(NKspace)+" "+to_string(rounds)+".txt");
	out<<"Agent id #"<<","<<"scores\n";
	for (vector<Agent>::iterator i = Agents.begin(); i != Agents.end(); i++)
	{
		out<<std::setprecision(15)<<i->id<<","<<i->score<<"\n";
	}
}

void output_round(int NKspace,int round,vector<int> rounds,vector<double> scr,vector<double> ag,vector<int> mc,vector<int> us,vector<double> pu,int search,int typeout,vector<double> same,vector<double> diff,char method,int k_val){
	string outtype="";
    string outmethod="";
	if(typeout==-9)  outtype ="baseline";
	if(typeout==-1)  outtype="minority";
	if(typeout==1)  outtype ="majority";
	if(typeout==0)  outtype ="Swap";
    if(typeout==100) outtype="morph_exp";
    if(typeout==-100) outtype="morph_raw";
    if(method=='m') outmethod="matrix";
    if(method=='c') outmethod="connections";
    if(method=='s') outmethod="info_swap";
    if(method=='b') outmethod="basline";
    if(method=='z') {outmethod="morph";}
	std::fstream out("8c-NK_space_"+to_string(k_val)+'_'+to_string(NKspace)+"_"+outtype+"_SH_"+to_string(search)+"_"+outmethod+".txt",std::ios::out | std::ios::binary);
	out<<"round,"<<"max score,"<<"avg score,"<<"Number of unique solutions,"<<"percent with max score,"<<"avg similiar species,"<<"avg different species,"<<"minority count"<<"\n";
	for (int i=0;i<=round; i++){
		out<<std::setprecision(15)<<rounds[i]<<","<<scr[i]<<","<<ag[i]<<","<<us[i]<<","<<pu[i]<<","<<same[i]<<","<<diff[i]<<","<<mc[i]<<"\n";
	}
}
void swapper(vector<Agent> &v,vector<int> &a,vector<int> &b){ //swaping function
        for(unsigned int i=0;i<a.size();i++){
            v[b[i]].species=v[a[i]].species;
            v[b[i]].tempstring=v[a[i]].binarystring;
            v[b[i]].tempscore=v[a[i]].score;
        }
    }
void swap_agents(vector<Agent> &v,int mode){//swaps minority agents by swaping information
        vector<int> minor;
        vector<int> permute;
        for (int i = 0; i < 100; ++i)
        {
            if(mode==1)
            {
                if(v[i].minority==1)
                {
                    minor.push_back(v[i].id);
                    v[i].minority=0; 
                }
            }
            if(mode==-1)
            {
                if(v[i].minority!=1)
                {
                    minor.push_back(v[i].id);
                    v[i].minority=0; 
                }
            }
        }
        permute=minor;
        std::random_shuffle ( permute.begin(), permute.end() );
        swapper(v,minor,permute);
        for(unsigned int i=0;i<minor.size();++i)
        {
        	v[permute[i]].Agent::agent_swap_interal_info(v[permute[i]]);
        }
    }
    

int main(int argc, char *argv[]) {
std::ios::sync_with_stdio(false); 
    //std::string convert4(argv[4]);
    int opt;
    int start=-1;
    int end=-1;
    int searchm=-1;
    double prob=-1;
    int condition=-10;
    int mode=-10;
    char method='q';
    while((opt = getopt(argc, argv, "s:e:S:c:m:M:P:")) != -1)  
        {  
            switch(opt)  
            {  
            case 's': start=atoi(optarg); break;
            case 'e':  end=atoi(optarg); break;
            case 'S':  
                searchm=atoi(optarg); 
                break;     
            case 'c':  
                condition=atoi(optarg); 
                break;  
            case 'm':
                mode=atoi(optarg);
                break;
            case 'M': method=*optarg; break;
            case 'P': prob=atof(optarg); break;
        }  
    }
    if(start<=-1){
        cout<<" start value must be 0 or greater\n";
        return 0;
    }
    if(end==-1){
        cout<<" end value must be given\n";
        return 0;
    }    
    if(end<start)
        {
        cout<<" end value must be greater than start\n";
        return 0;
    }  
    if(searchm==-1){
        cout<<"Search heuristic required i.e 0,1,2 or 3\n";
        return 0; 
    }
    if(method=='z'){
        cout<<"Morhping will be used\nIf not desired, pass -M q\n";
        if(prob==-1) {
            cout<<"using exponetial probability function in Morhping\n Pass # 0<=P<=1 for raw probability\n";
            condition=100;
        }
        else condition=-100;
    }
    else{
        cout<<"Morhping will not be used\n";
        if(abs(condition)!=1){
            if(condition==-9){
                cout<<"using baseline\n";
            }
            else if(condition==0) cout<<"using swapping function\n";
            else{
                cout<<"Condition for seeking must be either -1 or 1 or -9 ;\nMajority seeking is 1 and Minority seeking is -1, baseline is -9\n";
                return 0; 
            }
        }
        if(abs(mode)!=1 && condition!=-9 && mode!=0){
            cout<<"mode must be either -1 or 1;\nMatrix is 1 and connections is -1\n";
            return 0; 
        }
    }


    //cout<<argc<<" argc \n"<<endl;
    //for (int i=0;i<argc;++i) cout<<"argv"<<i<<" "<<argv[i]<<endl;
    /* old code for parsing cmdline flags
    if(convert3=="-S" && atoi(argv[4])<=3){
        searchm=atoi(argv[4]);
    }
    else{
        searchm=0;
    }

    if(argc>=6 && atof(argv[5])>0)
    {
    	if(atof(argv[5])>=0.0) 
    	{
    		prob=atof(argv[5]);
    		condition=0;
    		cout<<"using "<<prob<<endl;
    	} 
    	else
    	{
    		prob=0.05; 
    		condition=0;
    		cout<<"using default probability of 0.05 for species switch\n";
    	}
    }
    else
    {
    	cout<<"skipping species swap \n";
    	prob=-1;
    	if(argc==8)
            {
            mode=atoi(argv[7]);
            //mode ==1 uses agent_swap_con / matrix method
            //mode ==-1 uses agent_minority_swap / connection swap.
            condition=atoi(argv[6]);
            //conditon selects majority seeking 1 or minority seeking -1 or uses swapping 0
            //or -9 for baseline
            }
    }
    */
    int NKspace_num=start;
    vector <Agent> agent_array(::agentcounta);
    cout.setf(std::ios::fixed);
    cout.setf(std::ios::showpoint);
    cout.precision(15);
    vector <string> NKspacevals(::na);
    vector<double> NKspacescore(::na);
    open_space_string(NKspacevals);

    double max = 0.0;
    double avgscore = 0.0;
    int percuni=0;
    vector<double> maxscore(100);
    vector<int> maxround(100);
    vector<int> minoritycount(100);
    vector<double> avgscores(100);
    vector<char> type(100);
    vector<int> uniquesize(100);
    vector<double> percentuni(100);
    vector<double> elts(100);
    vector<double> same(100);
    vector<double> diff(100);
    //AB_list(type);
    //AB_random_list(type,0);

	int rounds = 0;
    int nums = 0;
    int mcount=0;
    vector<int> k_opts={1,5,10,15};
#pragma omp parallel for default(none) shared(end,searchm,::na,NKspacevals,NKspace_num,cout,prob,argc,condition,mode,method,::matrixa,k_opts) firstprivate(max,avgscore,percuni,maxscore,maxround,minoritycount,avgscores,uniquesize,percentuni,NKspacescore,rounds,mcount,agent_array,type,nums,elts,same,diff)   schedule(dynamic,end/4)
	for(int inksp=NKspace_num;inksp<end;++inksp){
		for (int opts = 0; opts < k_opts.size(); ++opts)
		{
			

		//srand(inksp+1);
    	cout<<"NK_space #:"<<inksp<<endl;
    	#ifdef _OPENMP
    	cout<<"\t thread #: "<<omp_get_thread_num()<<endl;
    	#endif
		if(inksp>0||opts>0)
		{
			agent_array.clear();
			NKspacescore.clear();
			maxscore.clear();
			maxround.clear();
			minoritycount.clear();
			avgscores.clear();
			type.clear();
	        uniquesize.clear();
	        percentuni.clear();
			agent_array.resize(100);
			NKspacescore.resize(100);
			elts.clear();
			elts.resize(100);
			maxscore.resize(100);
			maxround.resize(100);
			minoritycount.resize(100);
			avgscores.resize(100);
			type.resize(100);
	        uniquesize.resize(100);
	        percentuni.resize(100);
            same.clear();
            diff.clear();
            same.resize(100);
            diff.resize(100);
	        percuni=0;
			rounds = 0;
	    	nums = 0;
	    	mcount=0;
	    	max=0.0;
	        //::matrix[100][100]={};
	        //std::cin.get();
    	}
	//AB_list_seg(type);
	//AB_list_deseg(type);
   	//AB_part_25_list(type);
    	//AB_part_5_list(type);
	//AB_random_list(type,inksp);
    AB_list_permute(type,inksp);
    cout<<k_opts[opts]<<" "<<inksp<<endl;
	open_space_scores(inksp, NKspacescore,k_opts[opts]);
    for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
    {// assignment of information to agents, changes for each NK_Space
    	std::uniform_int_distribution<> ids(0,::na); //r
        std::random_device rdm;
        std::mt19937 gen(rdm());
        int rnums=ids(gen);
        //cout<<rnums<<"\n";
        i->Agent::agent_connections(nums, *i);
        i->species = type[nums];
        //cout<<i->id<<endl;
        //cout<<i->species<<endl;
        i->Agent::agent_change(rnums, *i, NKspacevals, NKspacescore);

        nums++;
    }
    agent_connections_replacement_number(agent_array,inksp);
	/*for(int i=0;i<100;++i){
		cout<<"new items\n";
		cout<<agent_array[i].id<<" "<<agent_array[i].species<<" "<<agent_array[i].connection_replace<<" \n";


	}*/

    for (; rounds < 100; ++rounds) {
    	elts.clear();
		elts.resize(100);
    	/*
		intial data collection before every round starts its run
		runs through all agents to find number of unique solutions, max score and percent unique
    	*/
    	if(rounds==0){
    		        	for (int i=0; i<::agentcounta;++i)
        		{// number of unqiue scores
        			elts[i]=agent_array[i].score;
        		}
        	std::sort(elts.begin(),elts.end());
        	auto ip=std::unique(elts.begin(),elts.end());
        	elts.erase(ip,elts.end());
        	uniquesize[rounds]=elts.size();

        max=0;
        for (int i = 0; i < ::agentcounta; ++i) 
        {
            //cout<<agent_array[i].score<<endl;
            if (max < agent_array[i].score) 
            {
                max = agent_array[i].score;
            }

        }
        percuni=0;
        
        for (int i = 0; i < 100; ++i)
        {// percent of agents with max score
            if(max==agent_array[i].score)
            {
                percuni++;
                //cout<<percuni<<" ";
            }

        }
        percentuni[rounds]=(percuni);
        }
        //cout<<"round #"<<rounds<<endl;


        //cout<<percentuni[rounds]<<endl;
        mcount = 0;
        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
        {
            i->Agent::agent_minority_status(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
                            agent_array[i->connections[2]], agent_array[i->connections[3]],
                            agent_array[i->connections[4]], agent_array[i->connections[5]],
                            agent_array[i->connections[6]], agent_array[i->connections[7]]);
            if (i->minority == 1) 
            {
                mcount++;
            }
            i->minority=0;
        }
        
        /*
		end of data collection
        */
        //cout<<mcount<<endl;
        minoritycount[rounds] = mcount;
       
        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
        {
            i->Agent::agent_exploit(*i, agent_array[i->connections[0]], agent_array[i->connections[1]],
                            agent_array[i->connections[2]], agent_array[i->connections[3]],
                            agent_array[i->connections[4]], agent_array[i->connections[5]],
                            agent_array[i->connections[6]], agent_array[i->connections[7]],prob,method);
            //cout<<i->id<<" \033[1;31mid\033[0m "<<i->species<<" \033[1;32mspecies\033[0m "<<" "<<i->mutate_flag<< "\n";
            //cout<<i->minority<<" minority"<<" \033[1;34mnew_string\033[0m "<<endl;
            
        }


        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) {
            i->Agent::agent_swap_interal_info(*i);
            //cout<<i->id<<" id \t"<<i->score<<" \033[1;31mnew_score\033[0m "<<i->tempscore<<" \033[1;32mold_score\033[0m "<<"\n";
            //cout<<i->binarystring<<" \033[1;34mnew_string\033[0m "<<i->tempstring<<" \033[1;33mold_string\033[0m "<<"\n";
        }


        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
        {
            if (i->flag == -1) 
            {
                i->Agent::agent_explore(*i, NKspacescore,searchm);
                //agent_array[i].agentexplore(agent_array[i],NKspace);
                //cout<<i->id<<" id \t"<<i->score<<" \033[1;31mnew_score\033[0m "<<i->tempscore<<" \033[1;32mold_score\033[0m "<<"\n";
                //cout<<i->binarystring<<" \033[1;34mnew_string\033[0m "<<i->tempstring<<" \033[1;33mold_string\033[0m "<<"\n";
            }
        }
        if(condition==-9 && method!='z')
        {
            method='b';
            //runs baseline so nothing here
        }
        else
        {

            //if argv>=6 and probability is not given all connection manipulations are skipped
    	        if(condition==0)
                {
                    swap_agents(agent_array,mode);
                    method='s';
                }
                else
                {
    	        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i) 
    	        {
    	           if (condition==1)
                   {

                        if (i->minority == 1) 
    	               {//major seeking
    	                   //cout<<i->id<<" id \n";
    	              
    	                   if(mode==1)
                            {//matrix
                                
                                method='m';
    	                       i->Agent::agent_swap_hack(agent_array,*i,
    	                                       agent_array[i->connections[0]],agent_array[i->connections[1]],
    	                                       agent_array[i->connections[2]], agent_array[i->connections[3]],
    	                                       agent_array[i->connections[4]], agent_array[i->connections[5]],
    	                                       agent_array[i->connections[6]], agent_array[i->connections[7]]);
    	                   }
                            if(mode==-1)
                            {// connection reassignment
                                
                                method='c';
    	                       i->Agent::agent_minority_swap(i->id,*i,
    	                                       agent_array[i->connections[0]],agent_array[i->connections[1]],
    	                                       agent_array[i->connections[2]], agent_array[i->connections[3]],
    	                                       agent_array[i->connections[4]], agent_array[i->connections[5]],
    	                                       agent_array[i->connections[6]], agent_array[i->connections[7]]);
    	                   }
                        }
    	            }
                    if (condition==-1)
                   {
                    
                        if (i->minority != 1) 
                       {//minor seeking
                           //cout<<i->id<<" id \n";
                      
                           if(mode==1)
                            {
                                
                                method='m';
                               i->agent_swap_hack(agent_array,*i,
                                               agent_array[i->connections[0]],agent_array[i->connections[1]],
                                               agent_array[i->connections[2]], agent_array[i->connections[3]],
                                               agent_array[i->connections[4]], agent_array[i->connections[5]],
                                               agent_array[i->connections[6]], agent_array[i->connections[7]]);
                           }
                            if(mode==-1)
                            {
                            
                                method='c';
                               i->agent_minority_swap(i->id,*i,
                                               agent_array[i->connections[0]],agent_array[i->connections[1]],
                                               agent_array[i->connections[2]], agent_array[i->connections[3]],
                                               agent_array[i->connections[4]], agent_array[i->connections[5]],
                                               agent_array[i->connections[6]], agent_array[i->connections[7]]);
                           }
                        }
                    }
    	        }
            }
        }
		
        // Next lines are for debugging matrix method.
        //memset(::matrix,0,sizeof(::matrix));
        //agent_array[99].matrix_print(agent_array);
	
	/* 
	 *
	 *
	 * start of end round data collection
	 *
	 *
	 */
    /*int mincount=0;
    for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
    {
        if(i->minority==1)
        mincount++;
    }
    minoritycount[rounds] = mincount;
    */
        double samespecies=0;
        for (vector<Agent>::iterator i = agent_array.begin(); i != agent_array.end(); ++i)
        {
            for (uint j = 0; j < i->connections.size(); ++j)
               {
                  if(i->species==agent_array[i->connections[j]].species) samespecies++;
               }   
        }
        same[rounds]=(samespecies/100);
        diff[rounds]=8-(samespecies/100);
	if(rounds>0)
	{
        	for (int i=0; i<::agentcounta;++i)
        		{// number of unqiue scores
        			elts[i]=agent_array[i].score;
        		}
        	std::sort(elts.begin(),elts.end());
        	auto ip=std::unique(elts.begin(),elts.end());
        	elts.erase(ip,elts.end());
        	uniquesize[rounds]=elts.size();

        max=0;
        for (int i = 0; i < ::agentcounta; ++i) 
        {
            //cout<<agent_array[i].score<<endl;
            if (max < agent_array[i].score) 
            {
                max = agent_array[i].score;
            }

        }
        percuni=0;
        
        for (int i = 0; i < 100; ++i)
        {// percent of agents with max score
            if(max==agent_array[i].score)
            {
                percuni++;
                //cout<<percuni<<" ";
            }

        }
        percentuni[rounds]=(percuni);
    	}
        //cout << minoritycount[rounds] << " \033[1;34mminority count for round\033[0m " << rounds << "\n";
        maxscore[rounds] = max;
        maxround[rounds] = rounds;
        avgscore = 0.0;
       
        for (int i = 0; i < ::agentcounta; ++i) 
        {
            avgscore += agent_array[i].score;
        }

        avgscores[rounds] = (avgscore / (::agentcounta));
        int eqflag=0;
        //output_connections(inksp,agent_array,rounds);
        //output_scores(inksp,agent_array,rounds);
	/*
	 *
	 *
	 * end of data collection
	 *
	 *
	 *
	 */
        //output_matrix_connections(inksp,agent_array,rounds);
        //output_connections(inksp,agent_array,rounds);
    	if((uniquesize[rounds]==1) || rounds>=70)
    	{// checks for break conditions
    		//cout<<fabs((avgscore/100.0)-eq)<<"fabs"<<endl;
    		eqflag=1;
    	}
        
    	if(eqflag==1){break;}
    	
    }
    //for (int i = 0; i < rounds; i++) {
        //cout << maxscore[i] << " \033[1;31mmax score for round\033[0m " << i << endl;
        //cout << avgscores[i] << " \033[1;32mavg score for round\033[0m " << i << "\n";
    
    //
    
    output_round(inksp,rounds,maxround,maxscore,avgscores,minoritycount,uniquesize,percentuni,searchm,condition,same,diff,method,k_opts[opts]);
	}
	}
    return 0;
}
