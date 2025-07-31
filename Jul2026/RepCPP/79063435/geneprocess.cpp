#include "geneprocess.h"
#include <math.h>
using namespace suchi_utils ;
extern "C" {
int write_png (char *outfile, unsigned char *indata, int ns, int nl) ; 
}

// note that the stacked array has the dimensions of uint (nl, totFrames, ns)
// therefore the first dimension in interferometer space, totFrames is framespace
// and ns is the samples perpendicular to interferometer dimension

GeneProcess::GeneProcess() 
{
    nsamps = ns_inst ;
    nlines = nl_inst ;
    ns = nsamps ;
    nl = nlines ;
    nb = 100 ;
    stacked = 0L ;
    numBBOthers = 0 ;
    hotbbraw = new unsigned short [ns*nl*nb] ;
    coldbbraw = new unsigned short [ns *nl *nb] ;
    hotbb = new float [ns * nl] ;
    coldbb = new float [ns * nl] ;
    wavenum = new float [160] ;
    wavelen = new float [160] ;
    bbPlanck = 0 ;
    totFrames = 0 ;
    array_start = 39 ;
    array_end = 233 ;
    leftFlag = false ;

    hotimag = 0 ;
	startl = 60 ;
	endl = 480 ;
	nptsSeg = 421 ;
    startFinWave = 0 ;
    stopFinWave = 159 ;
    nfinWaves = stopFinWave = startFinWave + 1 ;
    startwave = 0 ;
    stopwave = 160 ;
    radArray = 0;
    emissArr = 0 ;
    tempArr = 0 ;
    strcpy(outPref , "") ;
	segs = new float [ns_inst] ;

}

GeneProcess::~GeneProcess (){
    if (hotimag) {
        delete [] hotimag ;

        delete [] hotreal ;
        delete [] coldreal ;
        delete [] coldimag ;
        delete [] hotbbraw ;
        delete [] coldbbraw ;
    }
    if (radArray) delete [] radArray ;
    if (emissArr) delete [] emissArr ;
    if (tempArr) delete [] tempArr ;
}

void GeneProcess::setOutpref (char *s) {
    strcpy(outPref, s);
}

void GeneProcess::setSegParams(int array_start, int array_end, bool leftFlag)
{

    this->array_start = array_start ;
    this->array_end = array_end ;
    this->leftFlag = leftFlag ;
}

void GeneProcess::setSegParamsNew (float seg_yintcpt, float seg_slope, int npts, bool lFlag) {
    this->nptsSeg = npts ;
    this->seg_yintcpt = seg_yintcpt ;
    this->seg_slope = seg_slope ;
    this->leftFlag = lFlag ;
}



void GeneProcess::processIntColumn (float *outarr, int totFrames, int colno){


    int i, j ;
    // colno is actually the row in the stacked array, corresponding to the frame
    long stacked_index ;
    // get the 2d array out of the stacked array
    for (j=0; j<nsamps; j++){
        for (i=0; i<nlines; i++){
            stacked_index = long(j) * totFrames * nlines + colno * nlines + i ;
            outarr [j * nlines + i] = stacked [stacked_index] ;



        }
    }


}

/***
 * Read the wavelength file into two arrays, a wavenumber array and a wavelength array
 *
 */
void GeneProcess::readWaveFile (char * str){
    char instr[120] ;
    int i, count=0 ;
    float cwave, microns ;

    for (i=0; i<160; i++) {
        wavenum[i]= i ;
        wavelen[i]= i ;
    }

    // open the wavelength file for reading.....
    FILE *fil = fopen (str, "r") ;
    if (fil == NULL) {
        printf ("Problem with wavelength file : %s\r\n", str) ;
        return ;
    }

    while (1) {
        fgets (instr, 120, fil) ;
        if (strlen (instr) < 5 || feof(fil)) break ;
        sscanf (instr, "%d %f %f", &i, &cwave, &microns) ;
        if (i > 159) break ;
        wavenum [i] = cwave ;
        wavelen[i] = microns ;
    }

    fclose (fil) ;

}


/// Procedure to read an ascii file. Each line contains an absolute path and name of a blackbody file then a
/// comma, and then the degrees of the bb in Celsius
void GeneProcess::readBBFilesList (char * filename){
    int count = 0;
    char strline[420], *tmp ;

    this->bbNameList.clear() ;
    this->tempList.clear() ;
    /// File list is that which contains filename (with absolute path) and the
    FILE *fil = fopen (filename, "r") ;
    if (fil == NULL ) {
        printf ("Problem with bb file : %s\r\n", filename ) ;
        return ;
    }
    while (!feof(fil)){
        fgets (strline, 420, fil) ;
        if (strlen (strline) < 5)break ;
        tmp = strtok (strline, "," ) ;
        bbNameList.push_back(tmp) ;
        tmp = strtok (NULL, ",\r\n" ) ;
        tempList.push_back (atof(tmp)) ;
        printf ("%s, %f\r\n", bbNameList[count].c_str(), tempList[count]) ;
        count++ ;
    }
    fclose(fil) ;
    numBBOthers = count - 2 ;

}



void GeneProcess::readBBFiles (char* hotfile, char * coldfile, float hotT, float coldT){

    int i ;
    long npix = long (ns) * nl ;
    unsigned short *rawOthers = 0l ;
    FILE *fil ;
    if (numBBOthers)
    {
        rawOthers = new unsigned short [npix * nb] ;
        othersBB = new float [npix * numBBOthers] ;
    }

    fil = fopen (hotfile, "r") ;
    if (fil == NULL) {
        printf ("Problem with %s \r\n", hotfile) ;
        return ;
    }
    fread ((char *) hotbbraw, 2, npix * nb, fil) ;
    fclose (fil) ;
    fil = fopen (coldfile, "r") ;
    if (fil == NULL) {
        printf ("Problem with %s \r\n", coldfile) ;
        return ;
    }
    fread ((char *) coldbbraw, 2, npix * nb, fil) ;
    fclose (fil) ;


    flattenData (hotbbraw, hotbb, ns, nl, nb) ;
    flattenData (coldbbraw, coldbb, ns, nl, nb) ;

    for (i=0; i<numBBOthers; i++){
        fil = fopen (bbNameList[i+2].c_str(), "r") ;
        printf ("Problem with %s \r\n", bbNameList[i+2].c_str()) ;
        fread((char *)rawOthers, 2, npix * nb, fil) ;
        flattenData (rawOthers, (float *)&othersBB[ns*nl*i], ns, nl, nb) ;
        fclose (fil) ;
    }
    if (rawOthers) delete [] rawOthers ;
    this->processBB() ;
    bbNameList.clear() ;
    bbNameList.push_back(hotfile) ;
    bbNameList.push_back(coldfile) ;
    tempList.clear() ;
    tempList.push_back (hotT) ;
    tempList.push_back (coldT) ;
}

void GeneProcess::genBBCurves () {
    int i, j ;
    int numfiles = tempList.size() ;
    if (bbPlanck) delete [] bbPlanck ;
    stopwave = 0 ;
    startwave = 160 ;
    bbPlanck = new float [numfiles * 160 ] ;
    for (i=0 ; i<numfiles; i++) {
        for(j=0; j<160; j++){

            if (wavelen[j]<7. || wavelen[j]>15) {
                bbPlanck[i*160+j]=0. ;
                continue ;
            }
            if (j < startwave) startwave = j ;
            if (j > stopwave) stopwave = j ;
            bbPlanck[i*160+j] = bb2rad (tempList[i], wavelen[j]) ;
        }


    }

    nwaves = stopwave - startwave + 1 ;
}


void GeneProcess::readProcessFile (char *ifile) {

    float hotTemp, coldTemp ;
    char  *tmp, str[420], hotF[420], coldF[420] ;

    FILE *fil = fopen (ifile, "r") ;
    if (fil == NULL) {
        printf ("Problem with %s\r\n", ifile) ;
        return ;
    }
    // read the wavelength file
    fgets (str, 420, fil) ;
    this->readWaveFile (str) ;
    // get the hot bb file and its temperature
    fgets (str, 420, fil) ;
    tmp = strtok (str, ",") ;
    strcpy (hotF, tmp) ;
    tmp = strtok (NULL, "\r\n\0") ;
    hotTemp = atof (tmp) ;

    fgets (str, 420, fil) ;
    tmp = strtok (str, ",") ;
    strcpy (coldF, tmp) ;
    tmp = strtok (NULL, "\r\n\0") ;
    coldTemp = atof (tmp) ;

    readBBFiles (hotF, coldF, hotTemp, coldTemp) ;

    bbNameList.clear() ;
    bbNameList.push_back(hotF) ;
    bbNameList.push_back(coldF) ;
    tempList.clear() ;
    tempList.push_back (hotTemp) ;
    tempList.push_back (coldTemp) ;
    genBBCurves() ;

    // read in the outPref string
    fgets (str, 420, fil) ;
    strcpy (outPref, str) ;

    // read in the scan file
    fgets (str, 420, fil) ;
    strcpy(scanFile, str) ;
    fclose (fil) ;
}

void GeneProcess::processBB () {

    int i, colno, npts, npts0 ;
    float *tdat = new float [1024] ;

    //npts = (endl - startl + 1)*2 ;
	npts = nptsSeg * 2 ;
	npts0 =0;
    double *fftarr = new double [npts * 2] ;
    double *ddat = new double [npts * 2] ;


    hotreal = new float [npts * nsamps] ;
    coldreal = new float [npts * nsamps] ;
    hotimag = new float [npts * nsamps] ;
    coldimag = new float [npts * nsamps] ;

    for (colno=0; colno<this->nsamps; colno++){
        getProfileSegmented  (colno, tdat, &npts0, 0 ) ;
        for (i=0; i<npts; i++) {
            ddat [i*2]= tdat[i] ;
            ddat[i*2+1] = 0. ;
        }
		// fft hot bb
        fftcallComplex  (ddat, fftarr, npts, -1) ;
		// break up the fft result into real and complex results
        for (i=0; i<npts; i++) {
            hotreal [colno * npts + i] = fftarr[i*2] ;
            hotimag [colno * npts + i] = fftarr[i*2+1] ;
        }
        getProfileSegmented  (colno, tdat, &npts0, 1) ;
        for (i=0; i<npts; i++) {
            ddat [i*2]= tdat[i] ;
            ddat[i*2+1] = 0. ;
        }
		// fft cold bb
        fftcallComplex  (ddat, fftarr, npts, -1) ;
		// break up the fft result into real and complex results
        for (i=0; i<npts; i++) {
            coldreal [colno * npts + i] = fftarr[i*2] ;
            coldimag [colno * npts + i] = fftarr[i*2+1] ;
        }
    }
    delete [] fftarr ;
    delete [] ddat ;

}



void GeneProcess::geneFrame (unsigned short *indat, float *scene_frame){
    // we need profiles
    // bbhot is the blackbody planck curve for hot blackbody
    // bbcold for the cold one
    // these are produced by the genBBCurves routine
    // hotint is the actual interferogram for the hot bb
    // coldint is the cold interferogram
    int  npts ; //colno, i
    float *bbhot, *bbcold ;
    //npts = (array_end - array_start + 1) * 2 ;
	npts = nptsSeg * 2 ;

    /*
    float *hotreal = new float [npts] ;
    float *coldreal = new float [npts] ;

    float *hotimag = new float [npts] ;
    float *coldimag = new float [npts] ;
    */

    bbhot = bbPlanck ;
    bbcold = &bbPlanck[160] ;

    //QFile qf ("/home/harold/bbfile") ;
    //qf.open (QIODevice::WriteOnly) ;
    // get the fft of the hot interferogram
    // hot profile


    int N;
    {

        // new
        //    int N;
        N= npts ;
        //fftw_complex *in, *out;
		double *in, *out ;


        //    in = (fftw_complex*) &indat[0] ;
        //    out = (fftw_complex*) &outdat [0] ;

        //in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * 2);
        //out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * 2);
		in = new double [N*2] ;
		out = new double [N*2] ;

        //double *ddat = new double [npts * 2] ; // in
        //double *fftarr = new double [npts * 2] ; // out

        fftw_plan p;

        p = fftw_plan_dft_1d(N, (fftw_complex *) in, (fftw_complex *)out, FFTW_FORWARD, FFTW_ESTIMATE);

	#pragma omp parallel for \
		shared(indat, scene_frame, npts, bbhot, bbcold, in, out)
        for (int colno=0; colno < this->nsamps; colno++)
        {

            //        float *tdat = new float [512] ;
            float tdat[512];

            int i;
            int bbsamploc;

            //        float *scenereal = new float [npts] ;
            float scenereal[npts] ;

            //        float *sceneimag = new float [npts] ;
            float sceneimag[npts] ;

            //        float *g_real = new float [npts] ;
            float g_real[npts] ;

            //        float *g_im = new float [npts] ;
            float g_im[npts] ;

            //        float *off_real = new float [npts] ;
            float off_real[npts] ;

            //        float *off_im = new float [npts] ;
            float off_im[npts] ;

            float *scene_rad ;


            scene_rad = &scene_frame [160 * colno] ;
            for (i=0; i<160; i++) scene_rad [i] = 0. ;



            // now go through the resampled file getting the
            // profile for each frame


            // then get the profile from the scene frame
            getProfileSegmented  (indat, colno, tdat, npts) ;

            //qf.write ((char *) tdat, npts * 4) ;
            //qf.close() ;

            for (i=0; i<npts; i++) {
                //in[0][i*2]= tdat[i] ; //ddat
                //in[1][i*2+1]=0. ; // ddat
                in[i*2] = tdat[i] ; //ddat
                in[i*2+1] =0. ; // ddat
            }


            //        fftcallComplex (ddat, fftarr, npts, -1) ;

            // fftcallComplex

            fftw_execute(p); /* repeat as needed */

            for (i=0; i<npts; i++) {
                scenereal [i] = out[i*2] ; // fftarr
                sceneimag [i] = out[i*2+1] ; // fftarr
            }

            for (i=startwave; i<= stopwave; i++) {
                bbsamploc = colno * npts + i ;
                g_real[i] = (hotreal[bbsamploc]- coldreal[bbsamploc]) / (bbhot[i]- bbcold[i]) ;
                g_im[i] = (hotimag[bbsamploc]- coldimag[bbsamploc]) / (bbhot[i]- bbcold[i]) ;
                off_real[i] = (bbhot[i]*coldreal[bbsamploc]-bbcold[i]*hotreal[bbsamploc]) / (bbhot[i]- bbcold[i]) ;
                off_im[i] = (bbhot[i]*coldimag[bbsamploc]-bbcold[i]*hotimag[bbsamploc]) / (bbhot[i]- bbcold[i]) ;
                scene_rad[i] =sqrt(pow(scenereal[i]-off_real[i],2.) + pow(sceneimag[i]-off_im[i],2.)) /
                        sqrt(g_real[i]*g_real[i]+g_im[i]*g_im[i]) ;

            }

            //        delete [] fftarr ;
            //        delete [] ddat ;
            //delete [] scenereal ;
            //        delete [] sceneimag ;
            //        delete [] g_real ;
            //        delete [] g_im ;
            //        delete [] off_real ;
            //        delete [] off_im ;

        }

        fftw_destroy_plan(p);

        //fftw_free(in);
        //fftw_free(out);
		delete [] in ;
		delete [] out ;
    }

}



void GeneProcess::geneProfile (int colno, float *scene_rad, int bbnum){
    // we need profiles
    // bbhot is the blackbody planck curve for hot blackbody
    // bbcold for the cold one
    // these are produced by the genBBCurves routine
    // hotint is the actual interferogram for the hot bb
    // coldint is the cold interferogram
    int i, npts ;
    float *tdat = new float [512] ;

    float *bbhot, *bbcold ;


    // hot profile
    getProfileSegmented  (colno, tdat, &npts, 0 ) ;
    double *fftarr = new double [npts * 2] ;
    double *ddat = new double [npts * 2] ;
    float *hotreal = new float [npts] ;
    float *coldreal = new float [npts] ;
    float *scenereal = new float [npts] ;
    float *sceneimag = new float [npts] ;
    float *hotimag = new float [npts] ;
    float *coldimag = new float [npts] ;
    float *g_real = new float [npts] ;
    float *g_im = new float [npts] ;
    float *off_real = new float [npts] ;
    float *off_im = new float [npts] ;
    for (i=0; i<npts; i++) {
        ddat [i*2]= tdat[i] ;
        ddat[i*2+1] = 0. ;
        scene_rad[i] = 0. ;
    }

    bbhot = bbPlanck ;
    bbcold = &bbPlanck[160] ;

    // get the fft of the hot interferogram
    fftcallComplex  (ddat, fftarr, npts, -1) ;
    for (i=0; i<npts; i++) {
        hotreal [i] = fftarr[i*2] ;
        hotimag [i] = fftarr[i*2+1] ;
    }

    // cold bb
    getProfileSegmented  (colno, tdat, &npts, 1 ) ;
    for (i=0; i<npts; i++) {
        ddat [i*2 ]= tdat[i] ;
        ddat[i*2+1] = 0. ;
    }
    fftcallComplex (ddat, fftarr, npts, -1) ;
    for (i=0; i<npts; i++) {
        coldreal [i] = fftarr[i*2] ;
        coldimag [i] = fftarr[i*2+1] ;
    }


    // now go through the resampled file getting the
    // profile for each frame

    // then get the 1st other bb
    getProfileSegmented  (colno, tdat, &npts, bbnum ) ;
    for (i=0; i<npts; i++) {
        ddat [i*2]= tdat[i] ;
        ddat [i*2+1]=0. ;
    }
    fftcallComplex (ddat, fftarr, npts, -1) ;
    for (i=0; i<npts; i++) {
        scenereal [i] = fftarr[i*2] ;
        sceneimag [i] = fftarr[i*2+1] ;
    }

    for (i=startwave; i<= stopwave; i++) {
        g_real[i] = (hotreal[i]- coldreal[i]) / (bbhot[i]- bbcold[i]) ;
        g_im[i] = (hotimag[i]- coldimag[i]) / (bbhot[i]- bbcold[i]) ;
        off_real[i] = (bbhot[i]*coldreal[i]-bbcold[i]*hotreal[i]) / (bbhot[i]- bbcold[i]) ;
        off_im[i] = (bbhot[i]*coldimag[i]-bbcold[i]*hotimag[i]) / (bbhot[i]- bbcold[i]) ;
        scene_rad[i] =sqrt(pow(scenereal[i]-off_real[i],2.) + pow(sceneimag[i]-off_im[i],2.)) /
                sqrt(g_real[i]*g_real[i]+g_im[i]*g_im[i]) ;

    }


}


void GeneProcess::getProfile (int colno, float *data, int *npts, int bbnum){
    int i, j, number ;

    switch (bbnum) {
    case 0:
        for (i=0; i<nl; i++) {
            data[i] = hotbb[i*ns+colno] ;
        }
        break ;
    case 1:
        for (i=0; i<nl; i++) {
            data[i] = coldbb[i*ns+colno] ;
        }
        break ;
    default :
        number = bbnum - 2 ;
        for (i=0; i<nl; i++) {
            data[i] = othersBB[number * ns * nl + i*ns+colno] ;
        }
    }

    *npts = nl ;
}

void GeneProcess::getProfileSegmented (float *indat, int colno, float *data, int *npts){

    int i,npts1 ;
    float *tdat = new float [1024] ;

    for (i=0; i<nlines; i++){
		tdat[i] = indat[colno*nlines+i] ;
	}
	startl = segs [colno] ;
	endl = startl + nptsSeg -1. ;
	 
	npts1 = procColumn_noFFT (tdat, data, startl, endl, leftFlag) ;
	*npts = npts1 ;

	delete [] tdat ;

}


void GeneProcess::getProfileSegmented (int colno, float *data, int *npts, int bbnum){

	int pts ;
    float *tdat = new float [1024] ;

    getProfile (colno, tdat, npts, bbnum ) ;
	startl = segs [colno] ;
	endl = startl + nptsSeg - 1. ;

    pts = procColumn_noFFT (tdat, data, startl, endl, leftFlag) ;

    *npts = pts ;
	

    delete [] tdat ;
}

void GeneProcess::getProfileSegmented (unsigned short *indat, int colno, float *data, int npts){

    int i ;
    //float *tdat = new float [512] ;
    float startl, endl, tdat[1024] ;

	startl = segs[colno] ;
	endl = startl + nptsSeg - 1. ;

	
    for (i=0; i<nlines; i++){
        //tdat[i] = indat[i * nsamps + colno] ;
        tdat[i] = indat[colno * nlines + i] ;
    }

    //getProfile (colno, tdat, npts, bbnum ) ;
//    cout << "array_start " << array_start << endl;
//    cout << "array_end " << array_end << endl;

    procColumn_noFFT (tdat, data, startl, endl, leftFlag) ;

//    *npts = (array_end-array_start+1) * 2 ;

}


void GeneProcess::getProfileFFTMag (int colno, float *data, int *npts, int bbnum) {
    int i ;
    float *tdat = new float [512] ;
    double *fftarr, *ddat ;
    //ddat = new double [512] ;
    getProfile (colno, tdat, npts, bbnum) ;
    procColumn (tdat, data, array_start, array_end, leftFlag) ;
    //getProfileSegmented (colno, tdat, npts, bbnum ) ;
    /*
    fftarr = new double [*npts * 2] ;
    for (i=0; i<*npts; i++) ddat[i] = tdat[i] ;
    fftcall (ddat, fftarr, *npts, -1) ;
    getMag (fftarr, data, *npts) ;
    */

    //delete [] fftarr ;
    //delete [] ddat ;
    delete [] tdat ;

}

void GeneProcess::processCube (suchi_offsets *so) {
    int j, ib, icnt, ival ;
    int i;
    unsigned short *stackedArray ; //*iframe,
	unsigned char *midband  ;
	char pngfile [420] ;

    char outstr[120], outfile[420], outfile_hdr[420] ;
    sprintf (outfile,"%s_spec",outPref) ;
    sprintf (pngfile,"%s_midband.png",outPref) ;
    sprintf (outfile_hdr,"%s.hdr",outfile) ;
    FILE *fout=fopen (outfile, "w") ;
    if (fout==NULL) {
        printf ("Problem opening output : %s\r\n", outfile) ;
        return ;
    }



    stackedArray = so->outarr ;
    totFrames = so->totFrames ;

    findWaveRange () ;
    nfinWaves = stopFinWave - startFinWave + 1 ;

    // big array
    //    radArray = new float [totFrames * nsamps * nfinWaves] ;

    // small array
    radArray = new float [nsamps * nfinWaves] ;
	// midband is single band for png output 
	midband = new unsigned char [nsamps * totFrames] ;

    //    cout << "processCube :  ------------ " << array_end;

    for (int ii=0; ii< totFrames; ii++)
    {
        unsigned short *iframe;
        float *outFrame = new float [nsamps * 160] ;
        iframe = &stackedArray [long(ii) * nsamps * nlines] ;

        geneFrame (iframe, outFrame) ;

        for (j=0; j<nsamps; j++) {
            //qf.write ((char *)&outFrame[j*80+startFinWave], nfinWaves * 4) ;
            for (ib=stopFinWave; ib>=startFinWave; ib--) {
                // big array
                //                radArray [ii * nsamps * nfinWaves + j * nfinWaves + nfinWaves -1 - (ib - startFinWave)] = outFrame[j*80+ib] ;
                // small array
                radArray [j * nfinWaves + nfinWaves -1 - (ib - startFinWave)] = outFrame[j*160+ib] ;
            }
			// scale to byte but first normalize to 30 Watts/m2 sr cm
			ival = int(radArray[j*nfinWaves + 3*nfinWaves/4]/ 30. * 240.) ;
			if (ival < 0) ival = 0 ;
			if (ival > 255) ival = 255 ;
			midband[ii*nsamps+j] = (unsigned char) ival ;
        }
        if (!(ii % 10)) {
            printf ("processCube : processed line %d\r\n",ii) ;
            fflush(stdout);
            //            cout << "processCube : processed line " << ii << array_end;
        }

        // for small array
        fwrite ((char *) radArray, 4, nsamps * nfinWaves, fout) ;
        delete [] outFrame ;
    }
    // for big array
    //    fwrite ((char *) radArray, 4, nsamps * nfinWaves, fout) ;

    //fwrite ((char *) radArray, 4, totFrames * nsamps * nfinWaves, fout) ;
    fclose (fout) ;
	write_png (pngfile, midband, nsamps, totFrames) ;

    fout = fopen (outfile_hdr, "w") ;
    sprintf (outstr, "Envi = \r\ndescription = {\r\nTircis Radiance Cube.}\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "samples = %d\r\n", this->nsamps ) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "lines = %d\r\n", this->totFrames) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "bands = %d\r\n", this->nfinWaves) ;
    fputs (outstr, fout) ;
    sprintf (outstr, "header offset = 0\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "file type = ENVI Standard\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "data type = 4\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "interleave = bip\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength units = microns\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength  = {\r\n") ;
    fputs (outstr, fout) ;
    for (i=stopFinWave; i>= startFinWave; i--){
        sprintf(outstr, "%f", wavelen[i]) ;
        fputs (outstr, fout) ;
        if (i != startFinWave) {
            sprintf(outstr, ",\r\n") ;
            fputs (outstr, fout) ;
        }
    }
    sprintf(outstr, "}\r\n") ;
    fputs (outstr, fout) ;
    fclose (fout) ;

    printf ("Header file written : %s\r\n", outfile_hdr ) ;



    delete [] midband ;
}

void GeneProcess::readSegmentFile (char *infil) {
	int status, i, ival ;
    float val ;


	FILE *fin = fopen (infil, "r") ;
	if (fin == NULL) {
		printf ("Could not open %s\r\n", infil) ;
		return ;
	}
	for (i=0; i< ns_inst; i++) {
		fscanf (fin, "%d %f", &ival, &val) ;
		segs[i] = val ;
	}
	fclose (fin) ;
}


void GeneProcess::findWaveRange () {
    int i ;
    //QFile qf ("/home/harold/outwaves") ;
    char outstr[800] ;
    //qf.open (QIODevice::WriteOnly) ;
    for (i=1; i<stopwave; i++) {
        if (wavelen[i-1] >=7.5 && wavelen [i] < 7.5) {
            stopFinWave = i-1 ;
        }
        if (wavelen[i-1] >=12.5 && wavelen [i] < 12.5) {
            startFinWave = i ;
        }

    }

    for (i= stopFinWave; i >= startFinWave ;i--) {
        sprintf (outstr, "%f," , wavelen[i]) ;
        //qf.write (outstr) ;

    }
    //qf.close() ;
    nfinWaves = stopFinWave - startFinWave + 1 ;
    printf ("good wavelengths are %d to %d\r\n", startFinWave, stopFinWave) ;

}


void GeneProcess::calcEmissivity () {

    char outstr[120], outfile[420], outfile_hdr[420] ;
    int i, j, ib, baseBand =0 ;
    float diff, mindiff, refRadiance, bbRadiance, tempVal, radVal, waveLen, curWave, maxTemp ;
    mindiff = 1.E9 ;


    if (emissArr) delete [] emissArr ;
    if (tempArr) delete [] tempArr ;
    emissArr = new float [totFrames * nsamps * nfinWaves] ;
    tempArr = new float [totFrames * nsamps * nfinWaves] ;
    // find the baseband, we can assume this has a max emissivity of maxEmiss, all others will be
    // scaled by the radiance of baseband => pixel temp =>  emiss = measured radiance of new band / max radiance of new band
    for (i=startFinWave; i<= stopFinWave; i++){
        diff = fabs (wavelen[i]-11.8) ;
        if (diff < mindiff){
            baseBand = i ;
            mindiff = diff ;
        }
    }



    for (i=0; i< totFrames; i++) {

        for (j=0; j<nsamps; j++) {
            maxTemp = 0. ;
            // for the pixel, find the maximum temperature in wavelengths > 10 microns
            //for (ib=startFinWave; ib<stopFinWave; ib++){
            for (ib=stopFinWave; ib>=startFinWave; ib--) {
                waveLen = wavelen[ib] ;

                radVal = radArray [i * nsamps * nfinWaves + j * nfinWaves + ib - startFinWave] ;
                tempVal = suchi_utils::bb2temp (radVal, waveLen) ;
                tempArr [i * nsamps * nfinWaves + j * nfinWaves + ib - startFinWave] = tempVal ;
                if (waveLen<10.) continue ;
                if (tempVal > maxTemp) {
                    maxTemp = tempVal ;

                }

            }

            //refRadiance = radArray [i * nsamps * nfinWaves + j * nfinWaves + baseBand - startFinWave] ;
            //qf.write ((char *)&outFrame[j*80+startFinWave], nfinWaves * 4) ;
            //tempVal = suchi_utils::bb2temp (refRadiance, wavelen[baseBand]) ;
            for (ib=stopFinWave; ib>=startFinWave; ib--) {
                curWave = wavelen [ib] ;
                bbRadiance = suchi_utils::bb2rad (maxTemp, curWave) ;
                emissArr [i * nsamps * nfinWaves + j * nfinWaves + ib - startFinWave] =
                        radArray [i * nsamps * nfinWaves + j * nfinWaves + ib - startFinWave] / bbRadiance ;
            }

        }
        //qf.write ((char *)outFrame, nsamps * 80 * 4) ;
    }

    sprintf (outfile,"%s_emiss",outPref) ;
    sprintf (outfile_hdr,"%s.hdr",outfile) ;
    FILE *fout=fopen (outfile, "w") ;
    fwrite ((char *) emissArr, 4, totFrames * nsamps * nfinWaves, fout) ;
    fclose (fout) ;

    fout = fopen (outfile_hdr, "w") ;
    sprintf (outstr, "Envi = \r\ndescription = {\r\nTircis Emissivity Cube.}\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "samples = %d\r\n", this->nsamps ) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "lines = %d\r\n", this->totFrames) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "bands = %d\r\n", this->nfinWaves) ;
    fputs (outstr, fout) ;
    sprintf (outstr, "header offset = 0\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "file type = ENVI Standard\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "data type = 4\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "interleave = bip\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength units = microns\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength  = {\r\n") ;
    fputs (outstr, fout) ;
    for (i=stopFinWave; i>= startFinWave; i--){
        sprintf(outstr, "%f", wavelen[i]) ;
        fputs (outstr, fout) ;
        if (i != startFinWave)
            sprintf(outstr, ",\r\n") ;
        fputs (outstr, fout) ;
    }
    sprintf(outstr, "}\r\n") ;
    fputs (outstr, fout) ;
    fclose (fout) ;

    sprintf (outfile,"%s_temp",outPref) ;
    sprintf (outfile_hdr,"%s.hdr",outfile) ;

    fout=fopen (outfile, "w") ;
    fwrite ((char *) tempArr, 4, totFrames * nsamps * nfinWaves, fout) ;
    fclose (fout) ;

    fout = fopen (outfile_hdr, "w") ;
    sprintf (outstr, "Envi = \r\ndescription = {\r\nTircis Temperature Cube.}\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "samples = %d\r\n", this->nsamps ) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "lines = %d\r\n", this->totFrames) ;
    fputs (outstr, fout) ;
    sprintf(outstr, "bands = %d\r\n", this->nfinWaves) ;
    fputs (outstr, fout) ;
    sprintf (outstr, "header offset = 0\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "file type = ENVI Standard\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "data type = 4\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "interleave = bip\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength units = microns\r\n") ;
    fputs (outstr, fout) ;
    sprintf(outstr, "wavelength  = {\r\n") ;
    fputs (outstr, fout) ;
    for (i=stopFinWave; i>= startFinWave; i--){
        sprintf(outstr, "%f", wavelen[i]) ;
        fputs (outstr, fout) ;
        if (i != startFinWave)
            sprintf(outstr, ",\r\n") ;
        fputs (outstr, fout) ;
    }
    sprintf(outstr, "}\r\n") ;
    fputs (outstr, fout) ;
    fclose (fout) ;

}
