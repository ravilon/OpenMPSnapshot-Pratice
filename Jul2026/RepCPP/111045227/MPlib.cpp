#include "MPlib.h"

namespace MPlib
{
void data_Pruning(string source, string dest, int dataSize, int start_Idx)
{
int num_thread = omp_get_num_threads() - 1;
ifstream iData(source, ios::in);
vector<stringstream> oData(num_thread);
vector<string> raw_Data;
string head, line;
getline(iData, head);

//ܼƫŧi
vector<Mat> srcImages;
vector<vector<uchar>> srcVec;

for (int i = 0; i < dataSize; i++)
{
srcImages.push_back(Mat::zeros(Size(28, 28), CV_8UC1));
srcVec.push_back(vector<uchar>(28 * 28, 0));
}

#pragma omp parallel num_threads(num_thread) shared(start_Idx, raw_Data, srcImages, srcVec, oData)
{
//ŪϤ
#pragma omp master
{
int idx = 0;
while (iData.peek() != EOF && getline(iData, line) && idx < dataSize)
{
idx++;
#pragma omp task shared(srcImages) firstprivate(idx)
{
vector<uchar> img;
vector<string> readData;
readData = string_partition(line, ',');

for (int i = start_Idx; i < readData.size(); i++)
{
img.push_back(to_uchar(readData[i]));
}
Mat img_Mat(Size(28, 28), CV_8UC1, img.data());
cv::swap(srcImages[idx], img_Mat);
}
}
#pragma omp taskwait
iData.close();
}

//BzT
#pragma omp for nowait schedule(static)
for (int i = 0; i < srcImages.size(); i++)
{
Mat dst1, dst2;

//blur
medianBlur(srcImages[i], dst1, 3);
cv::swap(srcImages[i], dst1);

//sharp
equalizeHist(srcImages[i], dst2);
cv::swap(srcImages[i], dst2);

//Mat to 1-dim
srcImages[i] = srcImages[i].reshape(0, 1);
const uchar* p = srcImages[i].data;
vector<uchar> src(p, p + srcImages[i].cols);
std::swap(srcVec[i], src);
}

//XϤ
int thread_Id = omp_get_thread_num();

#pragma omp for schedule(static)
for (int i = 0; i < srcVec.size(); i++)
{
for (int j = 0; j < srcVec[i].size(); j++)
{
oData[thread_Id] << static_cast<unsigned>(srcVec[i][j]) << ",";
}
oData[thread_Id] << endl;
}
#pragma omp barrier

//XG
#pragma omp single
{
ofstream oImg(dest, ios::out);
oImg << head << endl;
for (int i = 0; i < num_thread; i++)
{
while (oData[i].peek() != EOF && getline(oData[i], line))
{
string temp_Img;
oData[i] >> temp_Img;
oImg << temp_Img << endl;
}
}
oImg.close();
}
}
}

unsigned char to_uchar(string data)
{
istringstream iData(data);
unsigned char val;
iData >> val;

return val;
}

vector<string> string_partition(const string &source, char delim)
{
vector<string> result;
stringstream ss;
ss.str(source);
string item;

while (getline(ss, item, delim))
{
result.push_back(item);
}
return result;
}

string to_word(int Val)
{
char tempWord[20];
sprintf_s(tempWord, 20, "%d", Val);

return string{ tempWord };
}
}