#pragma once
#include <string>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>
#include <math_constants.h>
#include <cuda_runtime.h>


void cudaJacobiSN(double k, double arg, double* result);
void cudaJacobiSnSimple(double k, double arg, double* result);
void cudaCompleteEllipticIntegralK(double k, double* result);
void cudaInCompleteEllipticIntegralFukushima(double theta, double k, double* result);
void cudaInCompleteEllipticIntegralSimple(double theta, double k, double* result);
void cudaCompleteEllipticIntegralSimple( double k, double* result);

class BlackHolePhysics {
public:
	BlackHolePhysics();
	static void InCompleteEllipticIntegralFukushima(double, double, double*);
	static void InCompleteEllipticIntegralSimple(double, double, double*);
	static void CompleteEllipticIntegralSimple( double, double*);
	static void CompleteEllipticIntegralK(double, double*);
	static void JacobiSN(double, double,  double*);
	static void JacobiSnSimple(double, double, double*);
	

};

