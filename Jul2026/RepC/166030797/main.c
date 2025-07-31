#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "omp.h"
#include "mycom.h"
#include "mynet.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

double f1(double x);

double f1(double x) {
double e = exp(x);
return 1;//0.5*(e+1.0/e);
}

double f2(double x);

double f2(double x) {
double e = exp(x);
return 1;//0.5*(e-1.0/e);
}

double f3(double x);

double f3(double x) {
double e = exp(x);
return 1;//0.5*(e-1.0/e);
}

double f(double x, double y, double z);

double f(double x, double y, double z) {
return f1(x) * f2(y) * f3(z);
}

int np, mp, nt, nl, ier, lp;
int np1, np2, np3, mp1, mp2, mp3;
char pname[MPI_MAX_PROCESSOR_NAME];
char sname[10] = "ex08b.p00";
MPI_Status Sta[100];
MPI_Request Req[100];
union_t buf;
double tick, t1, t2, t3;

FILE *Fi = NULL;
FILE *Fo = NULL;
int nx, ny, nz, fl = 1;
double xa, xb, ya, yb, za, zb;

int MyNetInit_1(int* argc, char*** argv, int* np, int* mp,
int* nl, char* pname, double* tick)
{
int i;
int provided;
i = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
if (i != 0){
fprintf(stderr,"MPI initialization error");
exit(i);
}

MPI_Comm_size(MPI_COMM_WORLD,np);
MPI_Comm_rank(MPI_COMM_WORLD,mp);
MPI_Get_processor_name(pname,nl);

*tick = MPI_Wtick();

sleep(1);

return 0;
}

int main(int argc, char *argv[]) {
int i, i1, i2, j, j1, j2, k, k1, k2;
double s, p, x, y, z, hx, hy, hz, hxyz;
sscanf(argv[1], "%d", &nt);
MyNetInit_1(&argc, &argv, &np, &mp, &nl, pname, &tick);
fprintf(stderr, "Netsize: %d, process: %d, system: %s, tick=%12le\n", np, mp, pname, tick);
sleep(1);

sprintf(sname + 7, "%02d", mp);
ier = fopen_m(&Fo, sname, "wt");
if (ier != 0) mpierr("Protocol file not opened", 1);

if (mp == 0) {
ier = fopen_m(&Fi, "ex08b.d", "rt");
if (ier != 0) mpierr("Data file not opened", 2);
i = fscanf(Fi, "xa=%le\n", &xa);
i = fscanf(Fi, "xb=%le\n", &xb);
i = fscanf(Fi, "ya=%le\n", &ya);
i = fscanf(Fi, "yb=%le\n", &yb);
i = fscanf(Fi, "za=%le\n", &za);
i = fscanf(Fi, "zb=%le\n", &zb);
i = fscanf(Fi, "nx=%d\n", &nx);
i = fscanf(Fi, "ny=%d\n", &ny);
i = fscanf(Fi, "nz=%d\n", &nz);
i = fscanf(Fi, "fl=%d\n", &fl);
fclose_m(&Fi);
fprintf(stderr, "read is OK\n");
}

MPI_Bcast(&fl, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (np > 1) {
if (fl < 1) { // variant 1:
if (mp == 0) {
buf.ddata[0] = xa;
buf.ddata[1] = xb;
buf.ddata[2] = ya;
buf.ddata[3] = yb;
buf.ddata[4] = za;
buf.ddata[5] = zb;
buf.idata[8] = nx;
buf.idata[9] = ny;
buf.idata[10] = nz;
for (i = 1; i < np; i++) {
MPI_Isend(buf.ddata, 7, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, Req + i);
}
MPI_Waitall(np - 1, Req + 1, Sta + 1);
} else {
MPI_Recv(buf.ddata, 7, MPI_DOUBLE, 0, MY_TAG, MPI_COMM_WORLD, Sta + 0);
xa = buf.ddata[0];
xb = buf.ddata[1];
ya = buf.ddata[2];
yb = buf.ddata[3];
za = buf.ddata[4];
zb = buf.ddata[5];
nx = buf.idata[8];
ny = buf.idata[9];
nz = buf.idata[10];
}
} else { // variant 2:
if (mp == 0) {
buf.ddata[0] = xa;
buf.ddata[1] = xb;
buf.ddata[2] = ya;
buf.ddata[3] = yb;
buf.ddata[4] = za;
buf.ddata[5] = yb;
buf.idata[8] = nx;
buf.idata[9] = ny;
buf.idata[10] = nz;
}
MPI_Bcast(buf.ddata, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
if (mp > 0) {
xa = buf.ddata[0];
xb = buf.ddata[1];
ya = buf.ddata[2];
yb = buf.ddata[3];
za = buf.ddata[4];
yb = buf.ddata[5];
nx = buf.idata[8];
ny = buf.idata[9];
nz = buf.idata[10];
}
}
}

MPI_Barrier(MPI_COMM_WORLD);
//fprintf(stderr, "barrier %d\n",mp);
//fprintf(stderr, "thread %d nx %d\n",mp,nx);
//fprintf(stderr, "thread %d ny %d\n",mp,ny);
//fprintf(stderr, "thread %d nz %d\n",mp,nz);

// Next code:

if (np == 1) {
np1 = 1;
np2 = 1;
np3 = 1;
}
else {
/*s = sqrt((double)np) * ((double)nx) / ((double)ny);
np1 = floor(s); if (s>0.5+((double)np1)) np1++;
np2 = np / np1;
if (np1*np2!=np) {
if (nx>ny) {np1 = np; np2 = 1;} else {np1 = 1; np2 = np;}
}*/
//fprintf(stderr, "thread %d calculated dddddddddddd %d %d\n",mp, (int)(pow(np, 1/3)*nx/ny));
int d;
for (int i = pow(np, 1 / 3.0) * nx / ny; i > 0; i--) {
if (np % i == 0) {
d = i;
break;
}
}
//fprintf(stderr, "thread %d d %d\n",mp,d);
int q;
for (int i = pow(d, 1 / 3.0) * ny / nz; i > 0; i--) {
if (d % i == 0) {
q = i;
break;
}
}
//fprintf(stderr, "thread %d q %d\n",mp,q);
np1 = np / d;
np2 = d / q;
np3 = q;
}
//fprintf(stderr, "thread %d calculated %d %d %d\n",mp, np1, np2, np3);
mp3 = mp / np2;
mp2 = mp % np2 / np1;
mp1 = mp % np2 % np1;
//fprintf(stderr, "thread %d calculated__ %d %d %d\n",mp, mp1, mp2, mp3);
if (mp == 0) fprintf(stderr, "Grid=%dx%dx%d\n", np1, np2, np3);

fprintf(Fo, "Netsize: %d, process: %d, system: %s, tick=%12le\n", np, mp, pname, tick);
fprintf(Fo, "Grid=%dx%dx%d coord=(%d,%d,%d)\n", np1, np2, np3, mp1, mp2, mp3);
fprintf(Fo, "xa=%le xb=%le ya=%le yb=%le za=%le zb=%le nx=%d ny=%d ny=%d fl=%d\n", xa, xb, ya, yb, za, zb, nx, ny,
nz, fl);

t1 = MPI_Wtime();

hx = (xb - xa) / nx;
hy = (yb - ya) / ny;
hz = (zb - za) / nz;
hxyz = hx * hy * hz;

if (np1 == 1) {
i1 = 0;
i2 = nx - 1;
}
else {
i1 = mp1 * (nx / np1);
if (mp1 < np1 - 1) i2 = i1 + (nx / np1) - 1; else i2 = nx - 1;
}

if (np2 == 1) {
j1 = 0;
j2 = ny - 1;
}
else {
j1 = mp2 * (ny / np2);
if (mp2 < np2 - 1) j2 = j1 + (ny / np2) - 1; else j2 = ny - 1;
}

if (np3 == 1) {
k1 = 0;
k2 = nz - 1;
}
else {
k1 = mp3 * (nz / np3);
if (mp3 < np3 - 1) k2 = k1 + (nz / np3) - 1; else k2 = nz - 1;
}


int step = (k2 - k1) / nt;
int cur_proc = nt;
int num = 0;
if (step != 0)
omp_set_num_threads(nt);
else {
step = 1;
omp_set_num_threads(k2 - k1);
cur_proc = k2 - k1;
}

s = 0;
if (nt > 1) {
#pragma omp parallel
{   
int i_=i, i1_=i1, i2_=i2, j_=j, j1_=j1, j2_=j2, k_=k, k1_=k1, k2_=k2;
double x, y, z;
int l = k1_ + step * omp_get_thread_num();
int r = l + step;
double res = 0;
if (omp_get_thread_num() == cur_proc - 1)
r = k2_ + 1;

for (k_ = l; k_ < r; k_++) {
if (omp_get_thread_num() == 0)
fprintf(stderr, "worker %d  thread %d, l %d r %d tn %d res%lf\n", mp,omp_get_thread_num(), l, r, k_, res);	
z = za + (k_ * 1.0 + .5) * hz;
for (j_ = j1_; j_ <= j2_; j_++) {
y = ya + (j * 1.0 + .5) * hy;
for (i_ = i1_; i_ <= i2_; i_++) {
x = xa + (i_ * 1.0 + .5) * hx;
res = res + hxyz * f(x, y, z);
}
}
}
#pragma omp critical
{
s = s + res;
num++;
fprintf(stderr, "thread %d l %d r %d tn %d res%lf\n", mp, l, r, omp_get_thread_num(), res);
}

}
while (num < cur_proc);
} else {
for (k = k1; k <= k2; k++) {
z = za + (k * 1.0 + .5) * hz;
for (j = j1; j <= j2; j++) {
y = ya + (j * 1.0 + .5) * hy;
for (i = i1; i <= i2; i++) {
x = xa + (i * 1.0 + .5) * hx;
s = s + hxyz * f(x, y, z);
}
}
}
}

//fprintf(stderr, "thread %d calculated s %lf\n",mp, s);

t2 = MPI_Wtime();

t1 = t2 - t1;

if (np == 1)
t2 = 0;
else {
p = s;
MPI_Reduce(&p, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
t2 = MPI_Wtime() - t2;
}

t3 = t1 + t2;

if (mp == 0) fprintf(stderr, "t1=%le t2=%le t3=%le int=%le\n", t1, t2, t3, s);

fprintf(Fo, "i1=%d i2=%d j1=%d j2=%d\n", i1, i2, j1, j2);
fprintf(Fo, "t1=%le t2=%le t3=%le int=%le\n", t1, t2, t3, s);

ier = fclose_m(&Fo);

MPI_Finalize();
return 0;
}
