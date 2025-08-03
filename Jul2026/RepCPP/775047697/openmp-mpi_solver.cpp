#include "openmp-mpi_solver.h"
#include <set>
#include <stack>
#include <unordered_map>
#include <limits.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

vector<vector<int>> OpenMPMPISolver::GenerateAllCombinations(const vector<int>& places, const map<int, int>& demand, int capacity, int max_stops, Graph& graph) {
vector<vector<int>> routes;
int n = places.size();
#pragma omp parallel for
for (int i = 1; i < (1 << n); i++) {
vector<int> route;
int total_demand = 0;
unordered_map<int, int> place_count;
bool invalid = false;
for (int j = 0; j < n; j++) {
if (i & (1 << j)) {
int place = places[j];
route.push_back(place);
total_demand += demand.at(place);
place_count[place]++;
if (total_demand > capacity || place_count[place] > max_stops || !graph.verifyValidRoute(route)) {
invalid = true;
break;
}
}
}
if (!invalid) {
#pragma omp critical
routes.push_back(route);
}
}
return routes;
}

bool OpenMPMPISolver::coversAllCities(const vector<vector<int>>& combination, const vector<int>& places) {
set<int> uncoveredCities(places.begin(), places.end());
for (const auto& route : combination) {
for (int city : route) {
uncoveredCities.erase(city);
}
}
return uncoveredCities.empty();
}

void OpenMPMPISolver::distributeWork(const vector<vector<int>>& routes, int& start, int& end, MPI_Comm comm) {
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);
int routeCount = routes.size();
int routesPerProcess = routeCount / size;
int remainder = routeCount % size;
start = rank * routesPerProcess + min(rank, remainder);
end = start + routesPerProcess + (rank < remainder ? 1 : 0);
}

void OpenMPMPISolver::FindBestCombination(const vector<vector<int>>& routes, vector<vector<int>>& currentCombination, int index, const vector<int>& places, int& bestCost, vector<vector<int>>& bestCombination, Graph& graph, MPI_Comm comm) {
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);
int start, end;
distributeWork(routes, start, end, comm);
#pragma omp parallel
{
stack<pair<int, int>> stack;
stack.push(make_pair(index, 0));
int localBestCost = INT_MAX;
vector<vector<int>> localBestCombination;
while (!stack.empty()) {
auto [i, option] = stack.top();
stack.pop();
if (option == 0) {
if (coversAllCities(currentCombination, places)) {
int totalCost = 0;
#pragma omp parallel for reduction(+:totalCost)
for (const auto& route : currentCombination) {
totalCost += graph.calculateRouteCost(route);
}
if (totalCost < localBestCost) {
#pragma omp critical
{
if (totalCost < localBestCost) {
localBestCost = totalCost;
localBestCombination = currentCombination;
}
}
}
continue;
}
}
if (option == 0 && i >= start && i < end && static_cast<size_t>(i) < routes.size()) {
currentCombination.push_back(routes[i]);
stack.push(make_pair(i, 1));
stack.push(make_pair(i + 1, 0));
} else if (option == 1) {
currentCombination.pop_back();
stack.push(make_pair(i + 1, 0));
}
}
#pragma omp critical
{
if (localBestCost < bestCost) {
bestCost = localBestCost;
bestCombination = localBestCombination;
}
}
}
MPI_Allreduce(MPI_IN_PLACE, &bestCost, 1, MPI_INT, MPI_MIN, comm);
if (bestCost == localBestCost) {
bestCombination = localBestCombination;
}
for (int i = 0; i < size; ++i) {
if (i == rank) {
for (const auto& route : bestCombination) {
int routeSize = route.size();
MPI_Send(&routeSize, 1, MPI_INT, 0, 0, comm);
MPI_Send(route.data(), routeSize, MPI_INT, 0, 0, comm);
}
} else if (rank == 0) {
vector<vector<int>> tempCombination;
for (size_t j = 0; j < bestCombination.size(); ++j) {
int routeSize;
MPI_Recv(&routeSize, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
vector<int> route(routeSize);
MPI_Recv(route.data(), routeSize, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
tempCombination.push_back(route);
}
if (coversAllCities(tempCombination, places)) {
int tempCost = 0;
for (const auto& route : tempCombination) {
tempCost += graph.calculateRouteCost(route);
}
if (tempCost < bestCost) {
bestCost = tempCost;
bestCombination = tempCombination;
}
}
}
}
}
