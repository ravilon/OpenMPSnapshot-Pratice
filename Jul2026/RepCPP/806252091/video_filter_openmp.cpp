#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;

// Structure to hold frame data and its processing results
struct ThreadData {
    Mat frame;
    Mat filtered_h1;
    Mat filtered_h2;
    int frameOrder;
};

// Function to apply a high-pass filter to an image
Mat applyHighPassFilter(const Mat image, const Mat filter) {
    Mat result;
    filter2D(image, result, -1, filter);
    return result;
}

int numThreads = 6;
queue<ThreadData> frameQueue;
queue<int> processedFrameQueue;
unordered_map<int, ThreadData> tabelaHash;
bool isReadingComplete = false;
bool isProcessingComplete = false;

// Function to process frames with high-pass filters
void processFrames() {
    while (true) {
        if (frameQueue.empty()) {
            #pragma omp flush(frameQueue)
            if (isReadingComplete)
                break;
        } else {
            ThreadData threadData;
            bool stop = false;

            #pragma omp critical
            {
                if (!frameQueue.empty()) {
                    threadData = frameQueue.front();
                    frameQueue.pop();
                } else {
                    stop = true;
                }
            }
            if (stop) {
                continue;
            }

            // Define high-pass filters
            Mat h1 = (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
            Mat h2 = (Mat_<int>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

            // Process the frame
            threadData.filtered_h1 = applyHighPassFilter(threadData.frame, h1);
            threadData.filtered_h2 = applyHighPassFilter(threadData.filtered_h1, h2);

            cout << "Processing Frame: " << threadData.frameOrder + 1 << " Out of " << numThreads << endl;

            #pragma omp critical
            {
                tabelaHash[threadData.frameOrder] = threadData;
                processedFrameQueue.push(threadData.frameOrder);
            }
        }
    }
    isProcessingComplete = true;
}

// Function to write processed frames to a video file
void writeVideo(VideoCapture& video, bool show) {
    VideoWriter videoWriter;
    Size frameSize;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps;
    bool isFirstFrame = true;
    int counter = 0;
    int i = 0;

    while (true) {
        if (processedFrameQueue.empty()) {
            if (isProcessingComplete)
                break;
        } else {
            #pragma omp critical
            {
                processedFrameQueue.pop();
            }

            counter++;
            while (counter % 100 == 0 || (isProcessingComplete && processedFrameQueue.empty())) {
                if (i > numThreads) {
                    break;
                }
                Mat filteredFrame = tabelaHash[i].filtered_h2;
                Mat frame;
                #pragma omp critical
                {
                    frame = tabelaHash[i].frame;
                    tabelaHash.erase(i);
                }

                if (counter == numThreads - 1) {
                    break;
                }
                if (isFirstFrame) {
                    frameSize = filteredFrame.size();
                    fps = video.get(CAP_PROP_FPS);
                    videoWriter.open("out/output_video.avi", codec, fps, frameSize);
                    isFirstFrame = false;
                }

                videoWriter.write(filteredFrame);

                if (show) {
                    imshow("Filtro h2", filteredFrame);
                    imshow("Frame original", frame);
                    if (waitKey(10) == 'q')
                        break;
                }
                i++;
                if (i >= counter) {
                    break;
                }
            }
            if (processedFrameQueue.empty() && isProcessingComplete)
                break;
        }
    }
    videoWriter.release();
}

// Function to read video frames and enqueue them for processing
void readVideo(VideoCapture& video) {
    Mat frame;
    int count = 0;
    cout << "Number of frames: " << numThreads << endl;

    while (video.read(frame)) {
        ThreadData threadData;
        threadData.frame = frame.clone();
        threadData.frameOrder = count;
        count++;

        #pragma omp critical
        {
            frameQueue.push(threadData);
        }
    }
    isReadingComplete = true;
}

// Main function to execute the program
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video-file-path> [show] [num_threads]" << endl;
        return -1;
    }

    bool show = (argc >= 3 && string(argv[2]) == "show");
    numThreads = (argc == 4) ? stoi(argv[3]) : 6;

    VideoCapture video(argv[1]);
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();

    // Parallel sections for reading, processing, and writing video frames
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                #pragma omp parallel num_threads(1)
                {
                    readVideo(video);
                }
            }

            #pragma omp section
            {
                #pragma omp parallel num_threads(1)
                {
                    writeVideo(video, show);
                }
            }
        }

        #pragma omp parallel num_threads(numThreads - 2)
        {
            processFrames();
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "OpenMP version execution time: " << duration.count() << " seconds" << endl;

    video.release();
    if (show) {
        destroyAllWindows();
    }

    return 0;
}
