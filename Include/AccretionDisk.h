#pragma once
#ifndef ACCRETIONDISK_H
#define ACCRETIONDISK_H
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <list>
#include <iomanip>
#include <unordered_map>
#include <map>
#include <stack>
#include <queue>
#include <deque>
#include <chrono>
#include <random>
#include <ctime>

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include "BlackHolePhysics.h"
#include <memory>
#include <string>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>
#include <math_constants.h>
#include <cuda_runtime.h>
#include "cpu_anim.h"

#define BMPDIM 1024


#define HANDLE_ERROR(x) \
{ \
	cudaError_t err = x; \
	if( cudaSuccess != err ) \
	{ \
		fprintf( stderr, \
		         "CUDA Error on call \"%s\": %s\n\tLine: %d, File: %s\n", \
		         #x, cudaGetErrorString( err ), __LINE__, __FILE__); \
		fflush( stdout );\
		exit( 1 ); \
	} \
}

struct DataBlock {
	double* dev_box_xy;//[indx] [indy][rPrimary][phiPrimary][rsPrimary][FoPrimary] [rSecundary] [phiSecundary] [rsSecundary] [FoSecundary][impactParameter][opacity][free]->11*BMPDIM*BMPDIM
	double* dev_BHDisk;//[ind_r] [ind_phi][wavelength]->1*BMPDIM*BMPDIM
	unsigned char* dev_bitmap;/// [indx] [indy] [Rcolor] [Gcolor] [Bcolor] [alphacolor]->4*BMPDIM*BMPDIM
	CPUAnimBitmap* bitmap;
};

//void computeCosAlpha(double, double*, double*, int);
//void seedParticles(double*, double*, int, const double, const double);
void seedParticles(int, double*, int, const double, const double);
void compute(int, double*, int, double, double);
void increment(int, double, double*, int, double, double, double, double, double, double, double);
void makeDisk(DataBlock*,double,double,double);
void generate_frame(DataBlock*, int);
void cleanup(DataBlock*);


class AccretionDisk {
public:
	AccretionDisk(void);
	AccretionDisk(const double&, const double&, const double&, const double&, const int&);
	~AccretionDisk();
	void playBMP(double,double);
	//void generate_frame(DataBlock* d, int);

public://variables
	DataBlock   data;
	CPUAnimBitmap  bitmap;

private://methods
	void createDiskBMP();

private://variables
	double M;
	double inclination;
	double innerRadius;
	double outerRadius;

	int nParticles;
	int nRadii = 100;
	double accretionRate = 1e-8;

	double maxFluxPrimary;
	double minFluxPrimary;
	double maxFluxSecundary;
	double minFluxSecundary;
	double maxWidth;
	double powerScale = 0.9;
	double incr = 1.0;// 5.0 * M_PI / 180.0;



	int DIM;
};

#endif