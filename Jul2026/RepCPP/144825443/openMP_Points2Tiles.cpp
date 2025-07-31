// openMP_Points2Tiles.cpp
// @author RilaShu
// Created at 2018/08/15
// at least millions points is useful.

#include "stdafx.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <direct.h>
#include <io.h>
#include <algorithm>
using namespace std;

// lat:(-90.0 ~ 90.0) , lng:(-180.0 ~ 180.0)
static double LatMin = -90.0;
static double LngMin = -180.0;
//normal tile resolution is 256*256
static int nResolution = 256;

//to store intermediate data
class Point {
public:
	double lat;//lat
	double lng;//lnt
};
class PointsTile {
public:
	int nLevel;
	int nRow;
	int nCol;
	vector <Point> Points;
};
//to store a common pointcount-tile data
class PointCount {
public:
	int nPosition;
	long nCount;
};
class PointCountTile {
public:
	int nLevel;
	int nRow;
	int nCol;
	vector <PointCount> pointCount;
};

class LevelCountMinMax {
	public:
		int nLevel;
		int nMin;
		int nMax;
};
//data
vector <Point> vcPoints; //all the points
vector <PointsTile> vcPointsTiles; //intermediate tiles
vector <LevelCountMinMax> vcLevelCountMinMax; //points count min-max in each leval

// the tile level to be cut, for a heatmap usage, 10 is suitable(city level)
// when intend to use high level, watch your memery 
int nTotalLevel = 0;
string sOutputFilePath = "";

void readPointsFromCSV(string pathInput, int nLatIndex, int nLngIndex, vector<Point> &vcPoints);
void cutTile(int nTotalLevel, vector<Point> points, vector <PointsTile> &vcPointsTiles);
void outputPointsTiles(string pathFile, vector <PointsTile> vcPointsTiles);
void transPointCountTiles(string pathFile, int nTotalLevel, vector <LevelCountMinMax> &vcLevelCountMinMax,int nResolution);

int main()
{
	clock_t start, end;
	start = clock();

	nTotalLevel = 10;
	sOutputFilePath = "D:\\VSProjects\\openMP_Points2Tiles\\dpTile";
	// read points from csv and cut
	readPointsFromCSV("pointsExample.csv", 11, 10, vcPoints);
	//outputPointsTiles(sOutputFilePath, vcPointsTiles); which could be choose
	transPointCountTiles(sOutputFilePath, nTotalLevel, vcLevelCountMinMax, nResolution);
	end = clock();
	cout << "Total time: " << (end - start) << " ms" << endl;
	getchar();
	return 0;
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

//the .csv file input part needs to be parallelled!
void readPointsFromCSV(string pathInput, int nLatIndex, int nLngIndex, vector<Point> &vcPoints) {
	ifstream inPointsCSV(pathInput, ios::in);
	string sLine = "";
	//Read by line, split by ','
	int nCount = 1;
	vector <string> vcFile;
	while (getline(inPointsCSV, sLine))
	{
		vcFile.push_back(sLine);
		if (vcFile.size() == 100000) {
#pragma omp parallel for
			for (int i = 0; i < vcFile.size(); i++)
			{
				vector <string> vcLine;
				SplitString(vcFile[i], vcLine, ",");
				Point tPoint;
				tPoint.lat = stod(vcLine[nLatIndex]); tPoint.lng = stod(vcLine[nLngIndex]);
				if (tPoint.lat < 90.0 && tPoint.lat > -90.0 && tPoint.lng < 180.0 && tPoint.lng > -180.0) {
#pragma omp critical
					vcPoints.push_back(tPoint);
				}
			}
			vcFile.clear();
		}
	}
#pragma omp parallel for
	for (int i = 0; i < vcFile.size(); i++)
	{
		vector <string> vcLine;
		SplitString(vcFile[i], vcLine, ",");
		Point tPoint;
		tPoint.lat = stod(vcLine[nLatIndex]); tPoint.lng = stod(vcLine[nLngIndex]);
		if (tPoint.lat < 90.0 && tPoint.lat > -90.0 && tPoint.lng < 180.0 && tPoint.lng > -180.0) {
#pragma omp critical
			vcPoints.push_back(tPoint);
		}
	}
	vcFile.clear();

	cout << "Points num " << vcPoints.size() << "\n";
	cutTile(nTotalLevel, vcPoints, vcPointsTiles);
	cout << "Cut tiles finished.\n";
}

// init the vcPointsTiles
void initTiles(int nTotalLevel, vector <PointsTile> &vcPointsTiles) {
	//Init tiles by function resize()
	int nTileTotalNum = 0;
	for (int k = 0; k < nTotalLevel; k++)
	{
		nTileTotalNum += int(pow(4.0, k + 1));
	}
	vcPointsTiles.resize(nTileTotalNum);
	//Init tiles details parallelly
	for (int i = 0; i < nTotalLevel; i++) {
		int nTileBaseNum = 0;
		for (int m = 1; m < i + 1; m++)
		{
			nTileBaseNum += int(pow(4.0, m));
		}
#pragma omp parallel for
		for (int m = 0; m < int(pow(2.0, i + 1)); m++) {
#pragma omp parallel for
			for (int n = 0; n < int(pow(2.0, i + 1)); n++) {
				int nTileNum = nTileBaseNum + m * int(pow(2.0, i + 1)) + n;
				vcPointsTiles[nTileNum].nLevel = i; vcPointsTiles[nTileNum].nRow = m; vcPointsTiles[nTileNum].nCol = n;
			}
		}
	}
}
//cut the points into vcPointsTiles
void cutTile(int nTotalLevel, vector<Point> points, vector <PointsTile> &vcPointsTiles) {
	clock_t start, end;
	start = clock();

	initTiles(nTotalLevel, vcPointsTiles);
	cout << "Init Finish.\n";
	for (int i = 0; i < nTotalLevel; i++) {
		int nTileBaseNum = 0;
		for (int m = 1; m < i + 1; m++)
		{
			nTileBaseNum += int(pow(4.0, m));
		}
		//Add points to tiles parallelly.
# pragma omp parallel for 
		for (int j = 0; j < points.size(); j++) {
			int nRow = int((points[j].lat - LatMin) / 180.0 * pow(2.0, i + 1));
			int nCol = int((points[j].lng - LngMin) / 360.0 * pow(2.0, i + 1));
			if (nRow >= 0 && nCol >= 0) {
				int nTileNum = nTileBaseNum + nRow * int(pow(2.0, i + 1)) + nCol;
# pragma omp critical
				vcPointsTiles[nTileNum].Points.push_back(points[j]);
			}
		}
	}
	end = clock();
	cout << "Tiles cut time: " << (end - start) << " ms" << endl;
}

//output intermediate data (which also useful)
void outputPointsTiles(string pathFile, vector <PointsTile> vcPointsTiles) {
	int flag = _mkdir(pathFile.c_str());
	string pathPointsTiles = pathFile + "\\tilepoints";
	flag = _mkdir(pathPointsTiles.c_str());
	for (int i = 0; i < nTotalLevel; i++)
	{
		string sTileFile = "\\" + to_string(i);
		string pathPointTileFile = pathPointsTiles + sTileFile;
		flag = _mkdir(pathPointTileFile.c_str());
	}
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < vcPointsTiles.size(); i++)
	{
		if (vcPointsTiles[i].Points.size() > 0)
		{
			ofstream output;
			string sFilePath = pathPointsTiles + "\\" + to_string(vcPointsTiles[i].nLevel) + "\\" + \
				to_string(vcPointsTiles[i].nLevel) + "_" + to_string(vcPointsTiles[i].nRow) + "_" + to_string(vcPointsTiles[i].nCol) + ".tilepoints";
			output.open(sFilePath, ios::trunc);
			for (int j = 0; j < vcPointsTiles[i].Points.size(); j++)
			{
				output << vcPointsTiles[i].Points[j].lat << "," << vcPointsTiles[i].Points[j].lng << "\n";
			}
			output.close();//关闭文件
		}
	}
	cout << "points tiles output finished\n";
}

/*void getFiles(string path, vector<string>& files, vector<string> &ownname)
{
	//文件句柄  x64 用 intptr_t
	intptr_t hFile = 0;
	//文件信息，声明一个存储文件信息的结构体  
	struct __finddata64_t fileinfo;
	string p;//字符串，存放路径
	if ((hFile = _findfirst64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
	{
		do
		{
			//如果是目录,迭代之（即文件夹内还有文件夹）  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				//文件名不等于"."&&文件名不等于".."
				//.表示当前目录
				//..表示当前目录的父目录
				//判断时，两者都要忽略，不然就无限递归跳不出去了！
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files, ownname);
			}
			//如果不是,加入列表  
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				ownname.push_back(fileinfo.name);
			}
		} while (_findnext64(hFile, &fileinfo) == 0);
		//_findclose函数结束查找
		_findclose(hFile);
	}
}*/

void transPointCountTiles(string pathFile, int nTotalLevel, vector <LevelCountMinMax> &vcLevelCountMinMax, int nResolution) {
	vcLevelCountMinMax.resize(nTotalLevel);
	for (int i = 0; i < nTotalLevel; i++)
	{
		vcLevelCountMinMax[i].nLevel = i;
		vcLevelCountMinMax[i].nMin = INT_MAX;
		vcLevelCountMinMax[i].nMax = -1;
	}
	//create file
	int flag = _mkdir(pathFile.c_str());
	string pathCountTiles = pathFile + "\\counttile";
	flag = _mkdir(pathCountTiles.c_str());
	for (int i = 0; i < nTotalLevel; i++)
	{
		string sTileFile = "\\" + to_string(i);
		string pathCountTileFile = pathCountTiles + sTileFile;
		flag = _mkdir(pathCountTileFile.c_str());
	}
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < vcPointsTiles.size(); i++)
	{
		if (vcPointsTiles[i].Points.size() > 0) {
			int *nPointCount = new int[nResolution * nResolution]();
			int nLevel = vcPointsTiles[i].nLevel; int nRow = vcPointsTiles[i].nRow; int nCol = vcPointsTiles[i].nCol;
			double dResolLat = 180.0 / (pow(2, nLevel + 1) * nResolution);
			double dResolLnt = dResolLat * 2;
#pragma omp parallel for
			for (int j = 0; j < vcPointsTiles[i].Points.size(); j++)
			{
				int nR = int((vcPointsTiles[i].Points[j].lat + 90.0 - (180.0 / pow(2, nLevel + 1))*nRow) / dResolLat);
				int nC = int((vcPointsTiles[i].Points[j].lng + 180.0 - (360.0 / pow(2, nLevel + 1))*nCol) / dResolLnt);
#pragma omp atomic
				nPointCount[nR * nResolution + nC] += 1;
			}
			int nCountMax = *max_element(nPointCount, nPointCount + nResolution * nResolution);
			int nCountMin = *min_element(nPointCount, nPointCount + nResolution * nResolution);
			if (nCountMax > vcLevelCountMinMax[nLevel].nMax) vcLevelCountMinMax[nLevel].nMax = nCountMax;
			if (nCountMin < vcLevelCountMinMax[nLevel].nMin) vcLevelCountMinMax[nLevel].nMin = nCountMin;
			ofstream output;
			string sFilePath = pathCountTiles + "\\" + to_string(nLevel) + "\\" + \
				to_string(nLevel) + "_" + to_string(nRow) + "_" + to_string(nCol) + ".counttile";
			output.open(sFilePath, ios::trunc);
#pragma omp parallel for
			for (int j = 0; j < nResolution * nResolution; j++)
			{
				if (nPointCount[j] > 0) {
					output << j << "#" << nPointCount[j] << "\n";
				}
			}
			output.close();
			delete[] nPointCount;
		}
	}
	// output minmax count for each level
	ofstream output;
	string sPath = pathCountTiles + "\\minmax.dat";
	output.open(sPath, ios::trunc);
	output << "level, mincount, maxcount\n";
	for (int i = 0; i < nTotalLevel; i++)
	{
		output << vcLevelCountMinMax[i].nLevel << "," << vcLevelCountMinMax[i].nMin << "," << \
			vcLevelCountMinMax[i].nMax << "\n";
	}
}
