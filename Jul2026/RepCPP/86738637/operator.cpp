///
/// \file src/test/operator.cpp
/// \brief Test suite to validate the operator classes
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#include "test/operator.hpp"

namespace alias
{
namespace test
{
namespace oper
{

bool ConvolutionTest()
{
    std::cout << "Convolution test : ";

    int picture_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix<int> picture(picture_data, 16, 4, 4);
#ifdef VERBOSE
    std::cout << std::endl << "Picture :" << picture;
#endif // VERBOSE

    int filter_data[9] = {1,1,1,1,-8,1,1,1,1};
    Convolution<int> filter(Matrix<int>(filter_data, 9, 3, 3));
#ifdef VERBOSE
    std::cout << std::endl << "Filter :" << filter.Data();
#endif // VERBOSE

    int result_data[16] = {5,6,3,-14,-12,0,0,-27,-24,0,0,-39,-71,-54,-57,-90};
    Matrix<int> result(result_data, 16, 4, 4);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<int> computed_result = filter * picture;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed_result;
#endif // VERBOSE

    bool test_result = Compare(result, computed_result);
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

void ConvolutionTime(size_t data_length, size_t filter_length)
{
    std::cout << "Convolution time : ";

    std::default_random_engine generator;
    generator.seed(123456789);
    std::uniform_int_distribution distribution(-100,100);
    size_t test_height = data_length*data_length;
    size_t test_width = data_length;

    int* A_data = new int[test_height*test_width]; // destroyed when A is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < test_height*test_width; ++i )
    {
        A_data[i] = distribution(generator);
    }
    alias::Matrix<int> A(A_data, test_height, test_width);

    int* f_data = new int[filter_length*filter_length]; // destroyed when u is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < filter_length*filter_length; ++i )
    {
        f_data[i] = distribution(generator);
    }
    alias::Convolution f(Matrix<int>(f_data, filter_length, filter_length));

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    f * A;

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end-start;

    std::cout << elapsed_time.count() << std::endl;
}

bool AbelTestBuild()
{
    std::cout << "Abel transform build test : ";

    double result_data[64] = {6.776429738438966L, 0.0L, 0.0L, 0.0L, 4.517406684027948L, 3.939543120718442L, 0.0L, 0.0L, 3.763966562404914L, 5.556977595779923L, 0.0L, 0.0L, 3.591663046625439L, 6.000000000000001L, 0.0L, 0.0L, 4.969510102469053L, 3.149603149604725L, 0.0L, 0.0L, 3.606742824180582L, 5.959865770300537L, 0.0L, 0.0L, 3.205256586605265L, 3.834537299556675L, 3.298484500494130L, 0.0L, 3.099690470710479L, 3.483314773547883L,4.000000000000001L, 0.0L, 4.145350631997715L, 4.681879964287850L, 0.0L, 0.0L, 3.281002697735889L, 4.151213335685217L, 2.742261840160418L, 0.0L, 2.973519495711604L, 3.146386743399045L, 4.783304297240559L, 0.0L,   2.889317474424724L, 2.954708629106140L, 3.291502622129181L, 2.000000000000000L, 3.959797974644668L, 5.091168824543142L, 0.0L, 0.0L, 3.191441739482032L, 3.783630828275116L, 3.394112549695429L, 0.0L, 2.907105848336467L, 2.993426761378061L, 3.487536283878574L, 1.697056274847714L, 2.828427124746190L, 2.828427124746191L, 2.828427124746190L, 2.828427124746190L};
    Matrix<double> result(result_data, 64, 16, 4);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << std::setprecision(16) << result;
#endif // VERBOSE

    AbelTransform<double> K(8, 64, 4);
#ifdef VERBOSE
    std::cout << "Computed result :" << std::setprecision(16) << K.Data();
#endif // VERBOSE

    bool test_result = Compare(result, K.Data());
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool AbelTestApply()
{
    std::cout << "Abel transform apply test : ";

    double target_data[4] = {8.01014622769739L, 4.88608973803579L, 9.63088539286913L, 4.88897743920167L};
    Matrix<double> target(target_data, 4, 4, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Target matrix :" << target;
#endif // VERBOSE

    AbelTransform<double> K(4, 16, 2);
#ifdef VERBOSE
    std::cout << std::endl << "Reduced Abel matrix :" << K.Data();
#endif // VERBOSE

    double result_data[16] = {35.822462949689758L, 36.137596788175713L, 36.137596788175713L, 35.822462949689758L, 39.241542045876251L, 36.476063612607106L, 36.476063612607106L, 39.241542045876251L, 23.951000180045721L, 41.068373881823092L, 41.068373881823092L, 23.951000180045721L,  21.864171789035566L, 35.353852846400969L, 35.353852846400969L, 21.864171789035566L};
    Matrix<double> result(result_data, 16, 16, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed = K*target;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed;
#endif // VERBOSE

    bool test_result = Compare(result, computed);
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool AbelTestApply2()
{
    std::cout << "Abel transform apply test 2 : ";

    double target_data[8] = {8.14723686393179L, 9.57506835434298L, 4.21761282626275L, 6.78735154857773L, 2.76922984960890L, 9.05791937075619L, 9.64888535199277L, 9.15735525189067L};
    Matrix<double> target(target_data, 8, 8, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Target matrix :" << target;
#endif // VERBOSE

    AbelTransform<double> K(8, 64, 4);
#ifdef VERBOSE
    std::cout << std::endl << "Reduced Abel matrix :" << K.Data();
#endif // VERBOSE

    double result_data[64] = {55.2091781708536L, 74.5257769312451L, 83.8743674549772L, 86.7125397023462L, 86.7125397023462L, 83.8743674549772L, 74.5257769312451L, 55.2091781708536L, 70.6454413490365L, 86.4511102292221L, 76.7416719077964L, 75.4773408312152L, 75.4773408312152L, 76.7416719077964L, 86.4511102292221L, 70.6454413490365L, 78.6024741678198L,  78.045056282331L, 74.5269613442854L, 79.2884777040187L, 79.2884777040187L, 74.5269613442854L,  78.045056282331L, 78.6024741678198L, 81.0097015312481L,  76.545008220707L, 78.5747410868872L, 81.2529885370636L, 81.2529885370636L, 78.5747410868872L,  76.545008220707L, 81.0097015312481L,  85.385381075195L, 96.4765836608538L, 91.7939939929231L, 86.6443107018285L, 86.6443107018285L, 91.7939939929231L, 96.4765836608538L,  85.385381075195L,  83.135371388058L, 94.9390754749439L, 100.915483980938L, 90.3207764186881L, 90.3207764186881L, 100.915483980938L, 94.9390754749439L,  83.135371388058L, 75.8977291309801L,  90.534286874125L,  96.228090669534L,  98.226749189087L,  98.226749189087L,  96.228090669534L, 90.534286874125L, 75.8977291309801L, 62.0541744543622L, 79.3796977339534L, 88.0866186934524L, 90.7834465749937L, 90.7834465749937L, 88.0866186934524L, 79.3796977339534L, 62.0541744543622L};
    Matrix<double> result(result_data, 64, 64, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed = K*target;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed;
#endif // VERBOSE

    bool test_result = Compare(result, computed);
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool AbelTestTransposed()
{
    std::cout << "Abel transform transpose test : ";

    double target_data[16] = {2.76922984960890L, 7.09364830858073L, 0.461713906311539L, 7.54686681982361L, 0.971317812358475L, 2.76025076998578L, 8.23457828327293L, 6.79702676853675L, 6.94828622975817L, 6.55098003973841L, 3.17099480060861L, 1.62611735194631L, 9.50222048838355L, 1.18997681558377L, 0.344460805029088L, 4.98364051982143L};
    Matrix<double> target(target_data, 16, 16, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Target matrix :" << target;
#endif // VERBOSE

    AbelTransform<double> K = AbelTransform<double>(4, 16, 2).Transpose();
#ifdef VERBOSE
    std::cout << std::endl << "Reduced Abel matrix :" << K.Data();
#endif // VERBOSE

    double result_data[4] = {140.158514836875L, 46.208797155969L, 30.5667725857632L, 139.337069897935L};
    Matrix<double> result(result_data, 4, 4, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed = K*target;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed;
#endif // VERBOSE

    bool test_result = Compare(result, computed);
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool AbelTestTransposed2()
{
    std::cout << "Abel transform transpose test 2 : ";

    double target_data[64] = {4.86791632403172L, 6.79135540865748L, 1.73388613119006L, 4.35858588580919L, 3.95515215668593L, 3.90937802323736L, 4.46783749429806L, 3.67436648544477L, 8.31379742839070L, 3.06349472016557L, 9.87982003161633L, 8.03364391602440L, 5.08508655381127L, 0.377388662395521L, 0.604711791698936L, 5.10771564172110L, 8.85168008202475L, 3.99257770613576L, 8.17627708322262L, 9.13286827639239L, 5.26875830508296L, 7.94831416883453L, 7.96183873585212L, 4.16799467930787L, 6.44318130193692L, 0.987122786555743L, 6.56859890973707L, 3.78609382660268L, 2.61871183870716L, 6.27973359190104L, 8.11580458282477L, 3.35356839962797L, 2.91984079961715L, 5.32825588799455L, 6.79727951377338L, 4.31651170248720L, 3.50727103576883L, 1.36553137355370L, 0.154871256360190L, 9.39001561999887L, 7.21227498581740L, 9.84063724379154L, 8.75942811492984L, 1.06761861607241L, 1.67168409914656L, 5.50156342898422L, 6.53757348668560L, 1.06216344928664L, 6.22475086001228L, 4.94173936639270L, 3.72409740055537L, 5.87044704531417L, 7.79051723231275L, 1.98118402542975L, 2.07742292733029L, 7.15037078400694L, 4.89687638016024L, 3.01246330279491L, 9.03720560556316L, 3.39493413390758L, 4.70923348517591L, 8.90922504330789L, 9.51630464777727L, 2.30488160211559L};
    Matrix<double> target(target_data, 64, 64, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Target matrix :" << target;
#endif // VERBOSE

    AbelTransform<double> K = AbelTransform<double>(8, 64, 4).Transpose();
#ifdef VERBOSE
    std::cout << std::endl << "Reduced Abel matrix :" << K.Data();
#endif // VERBOSE

    double result_data[8] = {619.382195686487L, 619.524561482076L, 337.443035970858L, 68.7231225286759L, 41.460354182176L, 264.814515081461L, 640.807956259755L, 604.932952016532L};
    Matrix<double> result(result_data, 8, 8, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed = K*target;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed;
#endif // VERBOSE

    bool test_result = Compare(result, computed);
    std::cout << (test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

void AbelTime(size_t pic_size)
{
    std::cout << "Abel time : ";

    AbelTransform<double> K(pic_size, pic_size*pic_size, pic_size/2);

    std::default_random_engine generator;
    generator.seed(123456789);
    std::normal_distribution<double> distribution(100.0,10.0);
    size_t test_height = pic_size;
    size_t test_width = 1;

    double* target_data = new double[test_height*test_width]; // destroyed when A is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < test_height*test_width; ++i )
    {
        target_data[i] = distribution(generator);
    }
    alias::Matrix<double> target(target_data, test_height, test_width);

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE
    int progress_step = std::max(1, (int) std::floor((double)(pic_size)/100.0));
    int step = 0;
    std::cout << std::endl;
#endif // VERBOSE
    for(size_t i = 0; i < pic_size; ++i)
    {
#ifdef VERBOSE
        if( i % progress_step == 0 )
            std::cout << "\r" << step++ << "/100";
#endif // VERBOSE
        K * target;
    }
#ifdef VERBOSE
    std::cout << "\r100/100" << std::endl;
#endif // VERBOSE

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end-start;

    std::cout << elapsed_time.count() << " seconds" << std::endl;
}

bool WaveletTest()
{
    std::cout << "Wavelet transform test : ";

    double picture_data[16] = {1.0L, 2.0L, 3.0L, 4.0L, 5.0L, 6.0L, 7.0L, 8.0L, 9.0L, 10.0L, 11.0L, 12.0L, 13.0L, 14.0L, 15.0L, 16.0L};
    Matrix<double> picture(picture_data, 16, 16, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Picture :" << picture;
#endif // VERBOSE

    Wavelet<double> daubechies_6_t = Wavelet<double>(daubechies, 6, true);

    double result_data[16] = {33.9999999996562L, 1.92870260709001L, 11.0375107155067L, -8.07873392230074L, 8.81797181064152L, -2.38323234580054L, 2.05073458214855e-12L, -2.11018414439074L, 7.58753530181676L, -1.93068105222996L, 1.00010971726405e-12L, 6.99973412565669e-12L, 1.30005450849069e-11L, 1.89994409094396e-11L, 2.49996690016019e-11L, 3.10010350723644e-11L};
    Matrix<double> result(result_data, 16, 16, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed_result = daubechies_6_t * picture;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed_result;
#endif // VERBOSE

    bool forward_test_result = Compare(result, computed_result);

    Wavelet<double> daubechies_6 = Wavelet<double>(daubechies, 6);

    double result_data_inverse[16] = {0.99999999983028L, 1.99999999983316L, 2.99999999982896L, 3.99999999983609L, 4.99999999986459L, 5.99999999983541L, 6.99999999982718L, 7.99999999982039L, 8.99999999983825L, 9.9999999998695L, 10.9999999998462L, 11.9999999998085L, 12.999999999849L, 13.9999999998423L, 14.9999999998291L, 15.9999999998298L};
    Matrix<double> result_inverse(result_data_inverse, 16, 16, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result_inverse;
#endif // VERBOSE

    Matrix<double> computed_result_inverse = daubechies_6 * computed_result;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result inverse :" << computed_result_inverse;
#endif // VERBOSE

    bool inverse_test_result = Compare(result_inverse, computed_result_inverse);

    std::cout << ( (forward_test_result && inverse_test_result) ? "Success" : "Failure") << std::endl;

    return (forward_test_result && inverse_test_result);
}

bool WaveletTest2()
{
    std::cout << "Wavelet transform test 2 : ";

    double picture_data[64] = {1.0L, 2.0L, 3.0L, 4.0L, 5.0L, 6.0L, 7.0L, 8.0L, 9.0L, 10.0L, 11.0L, 12.0L, 13.0L, 14.0L, 15.0L, 16.0L, 17.0L, 18.0L, 19.0L, 20.0L, 21.0L, 22.0L, 23.0L, 24.0L, 25.0L, 26.0L, 27.0L, 28.0L, 29.0L, 30.0L, 31.0L, 32.0L, 33.0L, 34.0L, 35.0L, 36.0L, 37.0L, 38.0L, 39.0L, 40.0L, 41.0L, 42.0L, 43.0L, 44.0L, 45.0L, 46.0L, 47.0L, 48.0L, 49.0L, 50.0L, 51.0L, 52.0L, 53.0L, 54.0L, 55.0L, 56.0L, 57.0L, 58.0L, 59.0L, 60.0L, 61.0L, 62.0L, 63.0L, 64.0L};
    Matrix<double> picture(picture_data, 64, 64, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Picture :" << picture;
#endif // VERBOSE

    Wavelet<double> daubechies_6_t = Wavelet<double>(daubechies, 6, true);

    double result_data[64] = {259.999999996210L, 1.88972633813596L, 79.7393276375432L, -75.2206811961887L, 59.0201381169291L, -16.9777986694379L, -0.618093103532931L, -33.7094359147837L, 44.4473787126933L, -12.4967671786793L, 4.71353511777295e-12L, 5.27133892092024e-11L, 1.00710773054402e-10L, 1.48670187272160e-10L, -0.297335850196028L, -19.8181685101491L, 35.2718872426618L, -9.53292938315740L, 2.05073458214855e-12L, 1.90255589060939e-11L, 3.59977603281436e-11L, 5.29644106350702e-11L, 6.99303948081820e-11L, 8.69186944640887e-11L, 1.03866693024202e-10L, 1.20844889650584e-10L, 1.37817757206449e-10L, 1.54792623163758e-10L, 1.71763936407388e-10L, 1.88737470097067e-10L, 2.05724326463042e-10L, -8.44073657741643L, 30.3501412073000L, -7.72272420890483L, 1.00010971726405e-12L, 6.99973412565669e-12L, 1.30005450849069e-11L, 1.89994409094396e-11L, 2.49996690016019e-11L, 3.10010350723644e-11L, 3.70004027416826e-11L, 4.30006030782693e-11L, 4.90009144371584e-11L, 5.50011147737450e-11L, 6.09997607980972e-11L, 6.70020705584307e-11L, 7.30002724935730e-11L, 7.90008058970670e-11L, 8.50024495235857e-11L, 9.10029829270798e-11L, 9.70017399737344e-11L, 1.03003605644858e-10L, 1.09002140646908e-10L, 1.15004006318031e-10L, 1.20999210651007e-10L, 1.27004851080414e-10L, 1.33003164037859e-10L, 1.39002365173724e-10L, 1.45003564711033e-10L, 1.51002321757687e-10L, 1.57001078804342e-10L, 1.63003166520070e-10L, 1.69009251038688e-10L, 1.74999570390355e-10L};
    Matrix<double> result(result_data, 64, 64, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result;
#endif // VERBOSE

    Matrix<double> computed_result = daubechies_6_t * picture;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << computed_result;
#endif // VERBOSE

    bool forward_test_result = Compare(result, computed_result);

    Wavelet<double> daubechies_6 = Wavelet<double>(daubechies, 6);

    double result_data_inverse[64] = {0.999999999073026L, 1.99999999905949L, 2.99999999900745L, 3.99999999898921L, 4.99999999915007L, 5.99999999911350L, 6.99999999911959L, 7.99999999913948L, 8.99999999919179L, 9.99999999926888L, 10.9999999992485L, 11.9999999992017L, 12.9999999992162L, 13.9999999992339L, 14.9999999992493L, 15.9999999992718L, 16.9999999992717L, 17.9999999992651L, 18.9999999992668L, 19.9999999992666L, 20.9999999992312L, 21.9999999991809L, 22.9999999991301L, 23.9999999990685L, 24.9999999990598L, 25.9999999990686L, 26.9999999990638L, 27.9999999990670L, 28.9999999990594L, 29.9999999990494L, 30.9999999990480L, 31.9999999990481L, 32.9999999990476L, 33.9999999990484L, 34.9999999990507L, 35.9999999990542L, 36.9999999990622L, 37.9999999990728L, 38.9999999990845L, 39.9999999990983L, 40.9999999990670L, 41.9999999990176L, 42.9999999989724L, 43.9999999989157L, 44.9999999988919L, 45.9999999988771L, 46.9999999988491L, 47.9999999988232L, 48.9999999988374L, 49.9999999988683L, 50.9999999988996L, 51.9999999989430L, 52.9999999989236L, 53.9999999988831L, 54.9999999988832L, 55.9999999988840L, 56.9999999989431L, 57.9999999990315L, 58.9999999990086L, 59.9999999989582L, 60.9999999991374L, 61.9999999991176L, 62.9999999990786L, 63.9999999990847L};
    Matrix<double> result_inverse(result_data_inverse, 64, 64, 1);
#ifdef VERBOSE
    std::cout << std::endl << "Expected result :" << result_inverse;
#endif // VERBOSE

    Matrix<double> computed_result_inverse = daubechies_6 * computed_result;
#ifdef VERBOSE
    std::cout << std::endl << "Computed result inverse :" << computed_result_inverse;
#endif // VERBOSE

    bool inverse_test_result = Compare(result_inverse, computed_result_inverse);

    std::cout << ( (forward_test_result && inverse_test_result) ? "Success" : "Failure") << std::endl;

    return (forward_test_result && inverse_test_result);
}

bool WaveletTest3()
{
    std::cout << "Wavelet transform test 3 : ";

    Matrix<double> picture("data/test/x_rand.data", 4096, 1);

    Wavelet<double> daubechies_6 = Wavelet<double>(daubechies, 6);

    Matrix<double> result("data/test/x_rand_iwt.data", 4096, 1);

    Matrix<double> computed_result = daubechies_6 * picture;

    bool inverse_test_result = Compare(result, computed_result);

    Wavelet<double> daubechies_6_t = Wavelet<double>(daubechies, 6, true);

    Matrix<double> result_forward("data/test/x_rand_iwt_fwt.data", 4096, 1);

    Matrix<double> computed_result_forward = daubechies_6_t * computed_result;

    bool forward_test_result = Compare(result_forward, computed_result_forward);

    std::cout << ( (inverse_test_result && forward_test_result) ? "Success" : "Failure") << std::endl;

    return (inverse_test_result && forward_test_result);
}

bool SplineTest()
{
    std::cout << "Spline transform test : ";

    double spline_data[64] = {0.0L, 4.959486314383995e-02L, 1.377484192837850e-01L, 3.126567175723750e-01L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 7.785729458557558e-02L, 1.686803386749540e-01L, 2.534623667394704e-01L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.999613434257024e-01L, 3.862675910539243e-05L, 2.981519222558311e-08L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.678944715457936e-01L, 3.106430890550390e-02L, 1.041219548702570e-03L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.999999952810338e-01L, 4.718962128869876e-09L, 4.00926840315234e-15L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.974907805763779e-01L, 2.505685860706064e-03L, 3.533562916033468e-06L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.999999999994236e-01L, 5.764607523027588e-13L, 5.368708464751047e-22L, 0.0L, 0.0L, 0.0L, 0.0L, 0.0L, 4.962088676984664e-01L, 3.786447751337924e-03L, 4.684550195624691e-06L, 0.0L};
    Matrix<double> expected_result(spline_data, 64, 8, 8);
#ifdef VERBOSE
    std::cout << std::endl << "Spline matrix :" << expected_result;
#endif // VERBOSE

    Spline<double> king = Spline<double>(8).Transpose();
#ifdef VERBOSE
    std::cout << std::endl << "Computed result :" << king.Data();
#endif // VERBOSE

    bool test_result = Compare(expected_result, king.Data());

    std::cout << ( test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}


bool BlurTest()
{
    std::cout << "Blur filter test : ";

    double blur_data[25] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Matrix<double> data(blur_data, 25, 5, 5);
    double filter_data[9] = {1,1,1,1,-8,1,1,1,1};
    Matrix<double> filter(filter_data, 9, 3, 3);
    double result_data[25] = {12,12,9,6,-12,-12,0,0,0,-30,-27,0,0,0,-45,-42,0,0,0,-60,-108,-78,-81,-84,-132};
    Matrix<double> expected_result(result_data, 25, 5, 5);

    Blurring<double> blur(filter, 5);

    bool test_result = Compare(expected_result, blur * data);

    std::cout << ( test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool FourierTest()
{
    std::cout << "Fourier test : ";

    alias::Fourier<double> fourier(8);
    alias::Matrix<double> signal(2, 6, 6);
    alias::Matrix<std::complex<double>> result_signal = fourier.FFT2D(signal);
    alias::Matrix<std::complex<double>> result_signal_inverted = fourier.IFFT2D(result_signal);
    std::cout << signal;
    std::cout << result_signal;
    std::cout << result_signal_inverted;

    return true;
}

bool AstroTest()
{
    std::cout << "Astro operator test : ";

    Matrix<double> x(std::string("data/test/x.data"), 4224, 1, double());

    Matrix<double> divx(std::string("data/test/divx.data"), 4224, 1, double());

    Matrix<double> E(std::string("data/test/E.data"), 4096, 1, double());

    Matrix<double> expected_result(std::string("data/test/res.data"), 4096, 1, double());

    AstroOperator astro(64, 64, 32, E, divx, false, WS::Parameters<double>());

    Matrix<double> computed_result = astro * x;

    bool test_result = Compare(expected_result, computed_result);

    std::cout << ( test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

bool AstroTestTransposed()
{
    std::cout << "Astro transposed operator test : ";

    Matrix<double> x(std::string("data/test/x_transposed.data"), 4096, 1, double());

    Matrix<double> divx(std::string("data/test/divx.data"), 4224, 1, double());

    Matrix<double> E(std::string("data/test/E.data"), 4096,1, double());

    Matrix<double> expected_result(std::string("data/test/res_transposed.data"), 4224,1, double());

    AstroOperator astro_transp(64, 64, 32, E, divx, true, WS::Parameters<double>());

    Matrix<double> computed_result = astro_transp * x;

    bool test_result = Compare(expected_result, computed_result);

    std::cout << ( test_result ? "Success" : "Failure") << std::endl;

    return test_result;
}

} // namespace oper
} // namespace test
} // namespace alias
