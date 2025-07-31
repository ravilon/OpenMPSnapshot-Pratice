//
// Created by davide on 06/10/22.
//

#ifndef N_GRAM_PARALLELNGRAM_H
#define N_GRAM_PARALLELNGRAM_H

#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <filesystem>

#include "JobsQueue.h"
#include "Utils.h"
#include "SharedHistogram.h"
#include "Other Solutions/HistogramCollector.h"
#include "Other Solutions/PartialHistogramsQueue.h"

void parallelNgramWords(std::string& file_name, std::string& out_folder_parallel, int n, int chunk_size, int num_threads);

void parallelNgramWords(std::string& file_name, std::string& out_folder_parallel, int n, int chunk_size, int num_threads){
    /* Parallel implementation of an algorithm to extract words N-grams histogram
     * This implementation is based on Producer-Consumers pattern */
    // double start = omp_get_wtime();
    std::ifstream infile(file_name);
    JobsQueue jobsQueue; // Queue in which jobs are stored
    SharedHistogram sharedHistogram; // Class in which the CONSUMERS write their private histogram at the end of their work
    // HistogramCollector histogramCollector;
    // PartialHistogramsQueue partialHistogramsQueue(num_threads);

    /* Each Consumer thread will produce a partial histogram, which is written in the end into a shared histogram
     * Finally the overall histogram is written to file. */
    #pragma omp parallel num_threads(num_threads) default(none) shared(infile, out_folder_parallel, jobsQueue, n, chunk_size, std::cout, sharedHistogram)
    {
        #pragma omp single nowait
        {
            /* PRODUCER THREAD: reads the input file and generates chunks of words (vector of strings)
             * which are enqueued and passed to CONSUMERS threads */

            std::vector<std::string> wordsLoaded;
            std::string border[n-1];

            std::string line;
            std::string processedLine;
            std::string tmp; // to store the last word in the line
            int counter;
            size_t pos;

            counter = 0;

            while (std::getline(infile, line)) {

                /* Remove from the line what is not a letter nor a space */
                std::remove_copy_if(
                        line.begin(),
                        line.end(),
                        std::back_inserter(processedLine),
                        std::ptr_fun<char&,bool>(&processInputChar));

                pos = processedLine.find(' '); // we look for spaces in order to distinguish words
                // std::cout << processedLine << std::endl;

                while (pos != std::string::npos) {
                    if(pos > 0) { // in such a way we ignore double spaces
                        wordsLoaded.push_back(processedLine.substr(0, pos));
                        counter += 1;
                    }

                    processedLine.erase(0, pos + 1);
                    pos = processedLine.find(' ');

                    if (counter >= chunk_size) { // every chunk of words is enqueued in the jobsQueue
                        jobsQueue.enqueue(wordsLoaded);

                        for(int i=0; i < n-1; i++) // we need to save the last n-1 words as they are needed for the next chunk
                            border[i] = wordsLoaded[chunk_size - n + 1 + i];

                        wordsLoaded.clear(); // clear the local current chunk

                        for(int i=0; i < n-1; i++) // add the last n-1 words
                            wordsLoaded.push_back(border[i]);
                        counter = n-1;
                    }
                }

                // check if there is another word at the end of the line
                if(!processedLine.empty()) {
                    tmp = processedLine.substr(0, processedLine.length() - 1);
                    if (!tmp.empty()) {
                        wordsLoaded.push_back(tmp);
                        counter += 1;
                    }
                }

                processedLine.clear();
            }

            if(counter > 0){ // last chunk of data (it will be smaller in size)
                jobsQueue.enqueue(wordsLoaded);
            }

            jobsQueue.producerEnd(); // notify that the PRODUCER work is done
            // std::cout << "PRODUCER has finished reading" << std::endl;
        }

        /* CONSUMER THREAD */
        std::vector<std::string> wordsChunk;

        std::map<std::string, int> partialHistogram; // histogram in which we store the ngrams as keys, and counters as values
        std::map<std::string, int>::iterator it;

        std::string ngram;
        size_t pos;

        while(!jobsQueue.done()){ // until the PRODUCER hasn't finished
            if(jobsQueue.dequeue(wordsChunk)){ // gather a job from the jobsQueue

                 //std::cout << "CONSUMER is working..." << std::endl;

                ngram = "";

                /* First ngram */
                for(int j=0; j < n; j++)
                    ngram += wordsChunk[j] + " ";

                it = partialHistogram.find(ngram);
                if(it != partialHistogram.end()) // add the ngram to the local (partial) histogram
                    it->second += 1;
                else
                    partialHistogram.insert(std::make_pair(ngram, 1));

                pos = ngram.find(' '); // detect where the first space is in order to locate the first word

                /* Following outputs */
                for(int i=n; i < wordsChunk.size(); i++){
                    ngram.erase(0, pos + 1); // we remove the first word in the previous output
                    ngram += wordsChunk[i] + " "; // and concatenate the last word for the following output

                    it = partialHistogram.find(ngram);
                    if(it != partialHistogram.end()) // add the ngram to the local (partial) histogram
                        it->second += 1;
                    else
                        partialHistogram.insert(std::make_pair(ngram, 1));

                    pos =  ngram.find(' ');
                }
            }
        }

        // SHARE the LOCAL HISTOGRAM
        sharedHistogram.mergeHistogram(partialHistogram); // 1st solution (shared histogram)
        // histogramCollector.addPartialHistogram(partialHistogram); // 2nd solution (collect the histograms and sum them only when it's time to write to file)
        /* // 3rd solution (use a binary reducer pattern)
        partialHistogramsQueue.enqueue(partialHistogram);

        std::map<std::string, int> partialHistogram1;
        std::map<std::string, int> partialHistogram2;

        while(!partialHistogramsQueue.done()){
            if(partialHistogramsQueue.dequeue(partialHistogram1, partialHistogram2)){

                for(auto& kv : partialHistogram1) {
                    it = partialHistogram2.find(kv.first);
                    if(it != partialHistogram2.end())
                        it->second += kv.second;
                    else
                        partialHistogram2.insert(std::make_pair(kv.first, kv.second));
                }

                partialHistogramsQueue.enqueue(partialHistogram2);
            }
        }
         */

    } // barrier

    /* SEQUENTIAL PHASE: write to output file the Total Histogram */
    sharedHistogram.writeHistogramToFile(out_folder_parallel + std::to_string(n) + "gram_outputParallelVersion.txt");
    // histogramCollector.writeHistogramToFile(out_folder_parallel + std::to_string(n) + "gram_outputParallelVersion.txt");
    // partialHistogramsQueue.writeHistogramToFile(out_folder_parallel + std::to_string(n) + "gram_outputParallelVersion.txt");
}

#endif //N_GRAM_PARALLELNGRAM_H
