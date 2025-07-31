#include "MandelbrotCalculatorOLD.h"

MandelbrotCalculatorOLD::MandelbrotCalculatorOLD(unsigned int* escapeCounts, int width, int height)
{
    this->width = width;
    this->height = height;
    this->escapeCounts = escapeCounts;
    sizeOfTheWorld = width*height*sizeof(unsigned int);
    temporaryResultSerialAVX = (double*)_aligned_malloc(4*sizeof(double), 32); //which boundary?32 or 64?
    temporaryResultsParallelAVX = (double**)malloc(height*sizeof(double*));
    for(int i = 0; i < height; i++)
        temporaryResultsParallelAVX[i] = (double*)_aligned_malloc(4*sizeof(double), 32);

    //initialize OpenCL
    GPUInit();
}

MandelbrotCalculatorOLD::~MandelbrotCalculatorOLD()
{
    _aligned_free(temporaryResultSerialAVX);
    for(unsigned int i = 0; i < height; i++)
        _aligned_free(temporaryResultsParallelAVX[i]);
    free(temporaryResultsParallelAVX);
}

std::string MandelbrotCalculatorOLD::readKernelSource(const std::string filename)
{
    std::ifstream inputFile(filename);

    if(!inputFile)
        return "";

    std::ostringstream buffer;
    buffer << inputFile.rdbuf();

    std::string sourceCode = buffer.str();

    inputFile.close();
    return sourceCode;
}

void MandelbrotCalculatorOLD::GPUInit()
{
    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    std::string kernelSource = readKernelSource(mandelbrotKernelFilename);
    if(kernelSource.empty())
    {
        qInfo() << "Couldn't read the kernel source file";
        return;
    }

    const char* tempSource = kernelSource.c_str();
    //qInfo() << tempSource;

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&tempSource, NULL, &err);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //if(err != CL_SUCCESS)
        //throw TranslateOpenCLError(err);

    qInfo() << "before first if";
    if (err != CL_SUCCESS)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char* log = new char[log_size];

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        qInfo() << "build not successful: " << log;

        std::string temp(log);
        delete[] log;
        return; //notify the GUI using signal/slot mechanism.The passed value should be a string.
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernelName, &err);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    // Number of work items in each local work group
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localSize), &localSize, NULL);

    // Number of total work items - localSize must be devisor

    globalSize = std::ceil((double)(width*height) / localSize) * localSize;

    // Create the worlds in device memory for our calculation
    escapeCountsGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, width*height*sizeof(unsigned int), NULL, &err);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    //set kernel arguments
    //mandelbrot(__global escapeCounts, int width, int height, double upperLeftX, double upperLeftY, double incrementX, double incrementY, int iterationCount)
    //only the first 3 args are set once

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &escapeCountsGPU);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    if(err != CL_SUCCESS)
        qInfo() << TranslateOpenCLError(err);
    clFinish(queue);
}

void MandelbrotCalculatorOLD::calculateCPUSerial(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
    double incrementX = (downRightX - upperLeftX) / (double)width;
    double incrementY = (upperLeftY - downRightY) / (double)height;

    //imaginaryValue and realValue are put before loops to avoid multiplication (that would lead to: imaginaryValue = upperLeftY + y*incrementY in every iteration, same for realValue)

    double imaginaryValue = upperLeftY;
    for(unsigned int y = 0; y < height; y++)
    {
        double realValue = upperLeftX;
        for(unsigned int x = 0; x < width; x++)
        {
            escapeCounts[y*width + x] = isMandelbrotNumber(realValue, imaginaryValue, numberOfIterations);
            realValue += incrementX;
        }
        imaginaryValue -= incrementY;
    }

    //this one is much slower because subtraction is replaced by multiplication for imaginaryValue
    /*for(int y = 0; y < canvasHeight; y++)
    {
        double imaginaryValue = upperLeftY - y*incrementY;
        for(int x = 0; x < canvasWidth; x++)
        {
            double realValue = upperLeftX + x*incrementX;
            escapeCounts[y*width + x] = isMandelbrotNumber(realValue, imaginaryValue, numberOfIterations);
        }
    }*/
}

void MandelbrotCalculatorOLD::calculateGPU(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
    //set kernel arguments
    //only the first 3 args are set once

    double incrementX = (downRightX - upperLeftX) / (double)width;
    double incrementY = (upperLeftY - downRightY) / (double)height;

    cl_int err = 0;

    err |= clSetKernelArg(kernel, 3, sizeof(double), &upperLeftX);
    err |= clSetKernelArg(kernel, 4, sizeof(double), &upperLeftY);
    err |= clSetKernelArg(kernel, 5, sizeof(double), &incrementX);
    err |= clSetKernelArg(kernel, 6, sizeof(double), &incrementY);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &numberOfIterations);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);
    clFinish(queue);

    //execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);


    err = clEnqueueReadBuffer(queue, escapeCountsGPU, CL_TRUE, 0, sizeOfTheWorld, escapeCounts, 0, NULL, NULL);
    if(err != CL_SUCCESS)
        throw TranslateOpenCLError(err);

    clFinish(queue);
}



void MandelbrotCalculatorOLD::calculateCPUParallel(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
    double incrementX = (downRightX - upperLeftX) / (double)width;
    double incrementY = (upperLeftY - downRightY) / (double)height;

    #pragma omp parallel for
    for(int y = 0; y < height; y++)
    {
        double imaginaryValue = upperLeftY - y*incrementY;
        double realValue = upperLeftX;
        for(unsigned int x = 0; x < width; x++)
        {
            escapeCounts[y*width + x] = isMandelbrotNumber(realValue, imaginaryValue, numberOfIterations);
            realValue += incrementX;
        }
    }
}

void MandelbrotCalculatorOLD::calculateAVXSerial(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{

    double incrementX = (downRightX - upperLeftX) / (double)width;
    double incrementY = (upperLeftY - downRightY) / (double)height;


    __m256d divergenceIterations, groupOfFour, imaginary, _upperLeftX, _four, _two, _incrementX, _secondaryReal, _secondaryImaginary;

    //double real, double imaginary
    //double secondaryReal = real
    //double secondaryImaginary = imaginary
    //(a+ib)^2 = a^2 - b^2 + i2ab

    _upperLeftX = _mm256_set1_pd(upperLeftX);
    _four = _mm256_set1_pd(4.0);
    _two = _mm256_set1_pd(2.0);
    _incrementX = _mm256_set1_pd(incrementX);


    int wholeParts = width / 4; //4 pixels are read at a time (AVX has 256-bit registers, it can fit 4 doubles).It is possible that canvas dimension won't be a multiple of 4

    for(unsigned int y = 0; y < height; y++)
    {
        __m256d _incrementor = _mm256_set_pd(3, 2, 1, 0); //will be used for calculating the real values with _incrementX.Should it be 3, 2, 1, 0 or 0, 1, 2, 3?

        //load imaginary value into the register (repeating it 4 times)
        double imaginaryComponent = upperLeftY - y*incrementY;
        imaginary = _mm256_set1_pd(imaginaryComponent);

        for(int x = 0; x < wholeParts*4; x += 4)
        {
            divergenceIterations = _mm256_setzero_pd();
            __m256d diverged = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); //when some part diverges, 0x0 will be written into that part.Initially, this register is set to all ones.This is accomplished by broadcasting -1 (0xFFF...FF) and casting it from integer variable to double variable
            //load real part (4 different numbers in total) into groupOfFour
            groupOfFour = _mm256_fmadd_pd(_incrementor, _incrementX, _upperLeftX);
            _secondaryImaginary = _mm256_setzero_pd();
            _secondaryReal = _mm256_setzero_pd();

            //evaluate divergence
            for(unsigned int i = 0; i < numberOfIterations; i++)
            {
                __m256d currentIteration = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)i));
                //Z^2=(a+ib)^2 = a^2 - b^2 + i2ab
                __m256d a2 = _mm256_mul_pd(_secondaryReal, _secondaryReal); //a^2
                __m256d b2 = _mm256_mul_pd(_secondaryImaginary, _secondaryImaginary); //b^2


                //check for divergence ((a2 + b2) > 4)
                __m256d moduloSquare = _mm256_add_pd(a2, b2); //a2 + b2
                __m256d comparisonMask = _mm256_cmp_pd(moduloSquare, _four, _CMP_LE_OQ);
                groupOfFour = _mm256_and_pd(groupOfFour, comparisonMask);
                //those that haven't diverged will be 0xFFF...FF in comparisonMask

                //there must be a register that holds the current iteration (repeats 4 times)
                //there must be a "bool" register that tells which parts have not diverged yet (1 = not diverged, 0 = diverged)
                //divergenceIterations &= diverged AND NOT(comparisonMask) - modify only when a flag in diverged is 1, and flag in comparisonMask is 0.
                //write 0 into the "bool" register
                //if all parts are zeroed out, stop calculating because all of them have diverged

                //divergenceIterations needs to get currentIteration at the part that diverged only if corresponding flag in the "bool" register is 1 and if corresponding flag in comparisonMask is 0
                divergenceIterations =_mm256_or_pd(divergenceIterations, _mm256_and_pd(currentIteration, _mm256_andnot_pd(comparisonMask, diverged))); //divergenceIterations = divergenceIterations AND (diverged AND NOT(comparisonMask))
                //set the corresponding "bool" flags: diverged = diverged AND comparisonMask.There might be problems if overflow happens.
                diverged = _mm256_and_pd(diverged, comparisonMask);

                //break if all 4 numbers have diverged
                if(_mm256_movemask_pd(diverged) == 0)
                    break;

                //end of divergence test

                __m256d tempReal = _mm256_add_pd(_mm256_sub_pd(a2, b2), groupOfFour); // a2 - b2 + r0
                _secondaryImaginary = _mm256_fmadd_pd(_mm256_mul_pd(_secondaryReal, _secondaryImaginary), _two, imaginary); //2*_secondaryReal*_secondaryImaginary + imaginary
                _secondaryReal = tempReal;
            }

            //store the results in escapeCounts
            _mm256_store_pd(temporaryResultSerialAVX, divergenceIterations);


            unsigned int first = *((unsigned int*)(temporaryResultSerialAVX));
            unsigned int second = *((unsigned int*)(temporaryResultSerialAVX + 1));
            unsigned int third = *((unsigned int*)(temporaryResultSerialAVX + 2));
            unsigned int fourth = *((unsigned int*)(temporaryResultSerialAVX + 3));

            //in which order should they be stored (little endian)?
            escapeCounts[y*width + x] = first;
            escapeCounts[y*width + x+1] = second;
            escapeCounts[y*width + x+2] = third;
            escapeCounts[y*width + x+3] = fourth;

            //prepare registers for loading new group of 4 real numbers
            //add _four to _incrementor
            _incrementor = _mm256_add_pd(_incrementor, _four);


        }

        if((wholeParts*4) != width)
        {
            //evaluate the rest one by one
            double realValue = upperLeftX + incrementX*(wholeParts*4);
            int counter = 0;
            for(int x = wholeParts*4; x < width; x++)
                escapeCounts[y*width + x] = isMandelbrotNumber(realValue + incrementX*(counter++), imaginaryComponent, numberOfIterations);
        }
    }

}

void MandelbrotCalculatorOLD::calculateAVXParallel(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{
    //every thread will have its own aligned array of 4 doubles whose purpose will be to extract data from AVX register.
    //problems might arise due to access to main memory to store results.Compare solution without per thread caching and the one without it.

    //temporaryResultsParallelAVX[]

    double incrementX = (downRightX - upperLeftX) / (double)width;
    double incrementY = (upperLeftY - downRightY) / (double)height;


    __m256d _upperLeftX,_four, _two, _incrementX;


    //double real, double imaginary
    //double secondaryReal = real
    //double secondaryImaginary = imaginary
    //(a+ib)^2 = a^2 - b^2 + i2ab

    _upperLeftX = _mm256_set1_pd(upperLeftX);
    _four = _mm256_set1_pd(4.0);
    _two = _mm256_set1_pd(2.0);
    _incrementX = _mm256_set1_pd(incrementX);


    int wholeParts = width / 4; //4 pixels are read at a time (AVX has 256-bit registers, it can fit 4 doubles).It is possible that canvas dimension won't be a multiple of 4

    #pragma omp parallel for
    for(int y = 0; y < height; y++)
    {
        __m256d divergenceIterations, groupOfFour, imaginary, _secondaryReal, _secondaryImaginary;
        __m256d _incrementor = _mm256_set_pd(3, 2, 1, 0); //will be used for calculating the real values with _incrementX.Should it be 3, 2, 1, 0 or 0, 1, 2, 3?


        double* temporaryResult = temporaryResultsParallelAVX[y];


        //load imaginary value into the register (repeating it 4 times)
        double imaginaryComponent = upperLeftY - y*incrementY;
        imaginary = _mm256_set1_pd(imaginaryComponent);

        for(int x = 0; x < wholeParts*4; x += 4)
        {
            divergenceIterations = _mm256_setzero_pd();
            __m256d diverged = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); //when some part diverges, 0x0 will be written into that part.Initially, this register is set to all ones.This is accomplished by broadcasting -1 (0xFFF...FF) and casting it from integer variable to double variable
            //load real part (4 different numbers in total) into groupOfFour
            groupOfFour = _mm256_fmadd_pd(_incrementor, _incrementX, _upperLeftX);
            _secondaryImaginary = _mm256_setzero_pd();
            _secondaryReal = _mm256_setzero_pd();

            //evaluate divergence
            for(unsigned int i = 0; i < numberOfIterations; i++)
            {
                __m256d currentIteration = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)i));
                //Z^2=(a+ib)^2 = a^2 - b^2 + i2ab
                __m256d a2 = _mm256_mul_pd(_secondaryReal, _secondaryReal); //a^2
                __m256d b2 = _mm256_mul_pd(_secondaryImaginary, _secondaryImaginary); //b^2


                //check for divergence ((a2 + b2) > 4)
                __m256d moduloSquare = _mm256_add_pd(a2, b2); //a2 + b2
                __m256d comparisonMask = _mm256_cmp_pd(moduloSquare, _four, _CMP_LE_OQ);
                groupOfFour = _mm256_and_pd(groupOfFour, comparisonMask);
                //those that haven't diverged will be 0xFFF...FF in comparisonMask

                //there must be a register that holds the current iteration (repeats 4 times)
                //there must be a "bool" register that tells which parts have not diverged yet (1 = not diverged, 0 = diverged)
                //divergenceIterations &= diverged AND NOT(comparisonMask) - modify only when a flag in diverged is 1, and flag in comparisonMask is 0.
                //write 0 into the "bool" register
                //if all parts are zeroed out, stop calculating because all of them have diverged

                //divergenceIterations needs to get currentIteration at the part that diverged only if corresponding flag in the "bool" register is 1 and if corresponding flag in comparisonMask is 0
                divergenceIterations =_mm256_or_pd(divergenceIterations, _mm256_and_pd(currentIteration, _mm256_andnot_pd(comparisonMask, diverged))); //divergenceIterations = divergenceIterations AND (diverged AND NOT(comparisonMask))
                //set the corresponding "bool" flags: diverged = diverged AND comparisonMask.There might be problems if overflow happens.
                diverged = _mm256_and_pd(diverged, comparisonMask);

                //break if all 4 numbers have diverged
                if(_mm256_movemask_pd(diverged) == 0)
                    break;

                //end of divergence test

                __m256d tempReal = _mm256_add_pd(_mm256_sub_pd(a2, b2), groupOfFour); // a2 - b2 + r0
                _secondaryImaginary = _mm256_fmadd_pd(_mm256_mul_pd(_secondaryReal, _secondaryImaginary), _two, imaginary); //2*_secondaryReal*_secondaryImaginary + imaginary
                _secondaryReal = tempReal;
            }

            //store the results in escapeCounts
            _mm256_store_pd(temporaryResult, divergenceIterations);


            unsigned int first = *((unsigned int*)(temporaryResult));
            unsigned int second = *((unsigned int*)(temporaryResult + 1));
            unsigned int third = *((unsigned int*)(temporaryResult + 2));
            unsigned int fourth = *((unsigned int*)(temporaryResult + 3));

            //in which order should they be stored (little endian)?
            escapeCounts[y*width + x] = first;
            escapeCounts[y*width + x+1] = second;
            escapeCounts[y*width + x+2] = third;
            escapeCounts[y*width + x+3] = fourth;

            //prepare registers for loading new group of 4 real numbers
            //add _four to _incrementor
            _incrementor = _mm256_add_pd(_incrementor, _four);
        }

        if((wholeParts*4) != width)
        {
            //evaluate the rest one by one
            double realValue = upperLeftX + incrementX*(wholeParts*4);
            int counter = 0;
            for(int x = wholeParts*4; x < height; x++)
                escapeCounts[y*width + x] = isMandelbrotNumber(realValue + incrementX*(counter++), imaginaryComponent, numberOfIterations);
        }
    }

}

unsigned int MandelbrotCalculatorOLD::isMandelbrotNumber(double real, double imaginary, unsigned int numberOfIterations)
{
    //coordinates are to be used in multithreaded implementation for determining which pixel should be coloured, thread safe container will receive a tuple (convergenceSpeed, coordinateX, coordinateY)
    double secondaryReal = 0;
    double secondaryImaginary = 0;

    for (unsigned int i = 0; i < numberOfIterations; i++)
    {

        //Z^2=(a+ib)^2 = a^2 - b^2 + i2ab
        double a2 = secondaryReal * secondaryReal; //a^2
        double b2 = secondaryImaginary*secondaryImaginary; //b^2

        //check divergence
        if((a2 + b2) > 4)
            return i;

        secondaryImaginary = 2*secondaryReal*secondaryImaginary + imaginary;
        secondaryReal = a2 - b2 + real;
    }
    return 0;
}


const char* MandelbrotCalculatorOLD::TranslateOpenCLError(cl_int errorCode)
{
    switch (errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
                                                                                                                //    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
                                                                                                                //    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70

    default:
        return "UNKNOWN ERROR CODE";
    }
}
