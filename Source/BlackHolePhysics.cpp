// One-stop header.
#include "BlackHolePhysics.h"


BlackHolePhysics::BlackHolePhysics() {
	;//constructor
}

void BlackHolePhysics::JacobiSN(float k, float arg,  float* result) {
	cudaJacobiSN(k, arg, result);
}
void BlackHolePhysics::CompleteEllipticIntegralK(float k, float* result) {
	cudaCompleteEllipticIntegralK( k, result);
}


void BlackHolePhysics::InCompleteEllipticIntegralFukushima(float theta, float k, float* result) {
	cudaInCompleteEllipticIntegralFukushima(theta, k, result);
}