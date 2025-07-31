#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>
using namespace std;
string substring(int start, int length, string strword) {
 string a = "";
 for (int i = start; i < length; i++) {
 a += strword[i];
 }
 return a;
}
12
int find_all(string sen, string word) {
 int wordLen = word.length();
 int start = 0;
 int endword = wordLen;
 int finLength = sen.length();
 int count = 0;
 string senWord = "";
 for (int i = 0; i <= sen.length(); i++) {
 if (endword > finLength) {
 break;

}
 senWord = substring(start, endword, sen);
 if ((senWord.compare(word)) == 0) {
 count++;
 if (endword > finLength) {
 break;

}

}
 start += 1;
 endword += 1;
return count; }
int display(string path, string word_to_search) {
 int totalCount = 0;
 string line;
 int count = 1;
 ifstream myfile(path);
 if (myfile.is_open()) {
 while (getline(myfile, line)) {
 totalCount += find_all(line, word_to_search);
 count++;

}

}
 else {
 cout << "File not open\n" << endl;

}
 return totalCount; }
int main() {
 string homeDir = getenv("HOME"); // Get the home directory
 string path = homeDir + "/pdc_files/"; // Path to the directory containing the text files
 string word_to_search = "";
 cout << "Enter a word to search: ";
 cin >> word_to_search;
 double time = omp_get_wtime();
 DIR* dirp = opendir(path.c_str()); // Open the directory
 struct dirent *dp;
 #pragma omp parallel shared(dirp) private(dp)
 {
 #pragma omp single
 {
 while ((dp = readdir(dirp)) != NULL) { // Iterate over all files in the directory
 string filename(dp->d_name);
 if (filename != "." && filename != = "..") { // Skip current directory and parent directory
entries
 string npath = path + filename;
 #pragma omp task
 {
 cout << "Total Count is " + to_string(display(npath, word_to_search)) <<
 " from file " + filename << endl;
 }
 }
 }
 }
 }
 closedir(dirp); // Close the directory
 time = omp_get_wtime() - time;
 cout << "Time is " + to_string(time);
 return 0;
}
