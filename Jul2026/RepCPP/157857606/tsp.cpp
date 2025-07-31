#include <iostream>
#include <algorithm> 
#include <fstream>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unordered_map>
#include <climits>
#include <vector>
#include <assert.h>
using namespace std;

// Store the city data
unordered_map<int,string> idToname;
unordered_map<int,float> idToXcord;
unordered_map<int,float> idToYcord;
float** distTable;

// Initialise the population size and the top fittest parents to be chosen
int n_popl = 3000;
int m_popl = 1000;
int num_iter = 1000;

// Declare global variables like numCities, initialChromosome(One of the permutations) & the global minimum of the cycle path
int numCities;
string initChromosome;
pair<float,string> minCycle = make_pair(INT_MAX,"");


///////////////////////////////////////////////////////////////////////
//////////////// Helper Functions /////////////////////////////////////

// Convert the cityId to the corresponding mapping
char getChar(int i){		/////a-z,0-8
	if (i<26)
		return char(97+i);
	else
		return char(22+i);
}

// Convert the mapping into the corresponding cityId
int getInt(char c){
	if (c>='a' && c<='z')
		return c-'a';
	else
		return c-'0'+26;
}

// Take the input from the file and fill the above data structures.
void input(char filename[]){
	string line;
	ifstream in;
	in.open(filename);
	if (!in.is_open()){
		cout << "File not found.\n";
		return;
	}
	else{
		string input_str = "";
		in >> input_str;
		in >> input_str;
		
		string str = "";
		in >> str;
		in >> str;
		numCities = stoi(str);
		initChromosome = "";

		in >> str;
		for (int i=0; i<numCities; i++){
			float c,d;
			string a = "";
			in >> a >> c >> d;
			idToname[i] = a;
			idToXcord[i] = c;
			idToYcord[i] = d;
			initChromosome += getChar(i);
		}
		return;
	}
}

// Make a distance table containing the distance between every pair of cities.
void fillDistances(){
	distTable = new float*[numCities];
	for (int i=0; i<numCities; i++){
		distTable[i] = new float[numCities];
		for (int k=0; k<numCities; k++){
			int j = 0;
			if (i==k)
				distTable[i][k] = 0.0;
			else{
				float n = sqrt(pow(idToXcord[i]-idToXcord[k],2) + pow(idToYcord[i]-idToYcord[k],2));
				distTable[i][k] = n;
			}
		}
	}
}

// Get the initial random population of size n_popl
vector<string> generateInitialPopulation(){
	vector<string> v;
	string s = initChromosome;
	for (int i =0; i<n_popl; i++){
		random_shuffle(s.begin(),s.end());
		v.push_back(s);
	}
	return v;
}

// Get the hamiltonian cycle distance, given the order of the cities in the chromosome
float getHamiltonianCycle(string s){
	float eucD = 0.0;
	assert(s.size()==numCities);
	for (int i=0; i<numCities; i++){
		int j = (i+1)%numCities;
		eucD += distTable[getInt(s[i])][getInt(s[j])];
	}
	return eucD;
}

bool compare(string s1,string s2){
	return (getHamiltonianCycle(s1)<getHamiltonianCycle(s2));
}

// Evaluate the fitness of the population and sort it.
float evalFitnessPopl(vector<string> &popl){
	sort(popl.begin(),popl.end(),compare);
	return getHamiltonianCycle(popl[0]);
}

// Obtain c1 using PMX on parents p1,p2, with probability 80%. Else c1 is p1
string performPMX(string p1, string p2){
	int c = rand()%100;
	if (c < 80){
		int l1 = rand()%p1.size();
		int l2 = rand()%p1.size();
		int low = l1 <= l2 ? l1 : l2;
		int high = l1 >= l2 ? l1 : l2;
		while (low<=high){
			char c = p2[low];
			for (int i = 0; i<p1.size(); i++){
				if (p1[i] == c){
					p1[i] = p1[low];
					p1[low] = p2[low];
					break;
				}
			}
			low++;
		}
	}
	return p1;
}

// Obtain c2 using GX on parents p1,p2, with probability 80%. Else c2 is p2
string performGX(string p1, string p2){
	int c = 10;//rand()%100;
	if (c < 80){
		string str = "";
		str += p2[0];
		int i1 = p1.find(str[0]);
		int i2 = 0;
		for (int i = 1; i<numCities; i++){
			int j1 = (i1+1)%numCities;
			int j2 = (i2+1)%numCities;
			if (distTable[getInt(p1[i1])][getInt(p1[j1])] <= distTable[getInt(p2[i2])][getInt(p2[j2])]){
				if (str.find(p1[j1]) == std::string::npos){
					str += p1[j1];
					i1 = j1;
					i2 = p2.find(str[i]);
				}
				else if (str.find(p2[j2]) == std::string::npos){
					str += p2[j2];
					i1 = p1.find(str[i]);
					i2 = j2;
				}
				else{
					while (str.find(p2[j2]) != std::string::npos){
						j2 = (j2+1)%numCities;
					}
					str += p2[j2];
					i1 = p1.find(str[i]);
					i2 = j2;
				}
			}
			else{
				if (str.find(p2[j2]) == std::string::npos){
					str += p2[j2];
					i1 = p1.find(str[i]);
					i2 = j2;
				}
				else if (str.find(p1[j1]) == std::string::npos){
					str += p1[j1];
					i1 = j1;
					i2 = p2.find(str[i]);
				} 
				else{
					while (str.find(p1[j1]) != std::string::npos){
						j1 = (j1+1)%numCities;
					}
					str += p1[j1];
					i1 = j1;
					i2 = p2.find(str[i]);
				}
			}
		}
		return str;
	}
	return p2;
}

// Mutate the given child, with probability 10%
void performMutation(string &c1){
	int c = rand()%100;
	if (c < 10){
		int l1 = rand()%c1.size();
		int l2 = rand()%c1.size();
		while (l2==l1){
			l2 = rand() % c1.size();
		}
		char c = c1[l1];
		c1[l1] = c1[l2];
		c1[l2] = c;	
	}
}

// The main function containing the main loop, with <num_iter> number of iterations
void solve(int numThreads){
	vector<string> initPopl = generateInitialPopulation();
	#pragma omp parallel for firstprivate(initPopl) num_threads(numThreads)
		for (int i = 0; i < num_iter; i++){
			// cout << "iteration : " << i << endl;
			
			// Obtains the localMin of the current population and update the global minima
			float localMin = evalFitnessPopl(initPopl);
			#pragma omp critical
				minCycle = min(make_pair(localMin,initPopl[0]),minCycle);
			
			// Initialise the new population
			vector<string> newPopulation;
			for (int i = 0; i<n_popl/2; i++){
				int l1 = rand() % m_popl;
				int l2 = rand() % m_popl;
				while (l2==l1){
					l2 = rand() % m_popl;
				}
				// Select 2 parents randomly from the first m_popl chromosomes
				string p1 = initPopl[l1];
				string p2 = initPopl[l2];

				// Obtain c1 by PMX on 2 parents, followed by mutation
				string c1 = performPMX(p1,p2);
				performMutation(c1);

				// Obtain c2 by GX on 2 parents, followed by mutation
				string c2 = performGX(p1,p2);
				performMutation(c2);

				// Add the 2 children to the new population
				newPopulation.push_back(c1);
				newPopulation.push_back(c2);
			}
			initPopl.clear();
			initPopl = newPopulation;
		}
}

// Write the output into the file
void output(string filename, float cost, string path){
	ofstream out(filename, ios::out);
	out << "DIMENSION : " << numCities << endl;
	out << "TOUR_LENGTH : " << to_string(cost) << endl;
	out << "TOUR_SECTION : " << path << endl;
	out << "-1\nEOF";
}	

////////////////////////////////////////////////////////////////
/////////////////// Main function //////////////////////////////

int main(int argc, char * argv[])
{
	// Read the command line arguments, filename and the number of threads
	char* infile = argv[1];
	char* outfile = argv[2];
	int numThreads = atoi(argv[3]);
	
	// Seed
	srand(time(NULL));
	
	// Call input(), which reads in the whole data
	input(infile);

	// Fill the distance matrix
	fillDistances();

	double start = omp_get_wtime();
	// Get the minima of the hamiltonian path
	solve(numThreads);
	
	// Print the smallest path and the path length
	cout << "Time taken : " << omp_get_wtime() - start << " s" << endl;
	cout << "Cost: " << minCycle.first << endl;
	string str = "";
	for (int i=0; i < (minCycle.second).size(); i++){
		str += idToname[getInt(minCycle.second[i])];
		str += " ";
	}
	cout << "Path : " << str << endl;
	
	// Write the output into the file
	output(outfile, minCycle.first, str);
	return 0;
}
