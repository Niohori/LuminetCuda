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
const float M_PI = 3.14159265358979323846;

void cudaJacobiSN(float k, float arg, float* result);
void cudaCompleteEllipticIntegralK(float k, float* result);
void cudaInCompleteEllipticIntegralFukushima(float theta, float k, float* result);
class BlackHolePhysics {
public:
	BlackHolePhysics();
	static void InCompleteEllipticIntegralFukushima(float, float, float*);
	static void CompleteEllipticIntegralK(float, float*);
	static void JacobiSN(float, float,  float*);
	

};

