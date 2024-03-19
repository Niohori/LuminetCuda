// One-stop header.
#include "cudastuff.h"


BlackHolePhysics::BlackHolePhysics() {
	;//constructor
}

void BlackHolePhysics::JacobiSN(double k, double arg,  double* result) {
	cudaJacobiSN(k, arg, result);
}

void BlackHolePhysics::JacobiSnSimple(double k, double arg, double* result) {
	cudaJacobiSnSimple(k, arg, result);
}
void BlackHolePhysics::CompleteEllipticIntegralK(double k, double* result) {
	cudaCompleteEllipticIntegralK( k, result);
}


void BlackHolePhysics::InCompleteEllipticIntegralFukushima(double theta, double k, double* result) {
	cudaInCompleteEllipticIntegralFukushima(theta, k, result);
}

void BlackHolePhysics::InCompleteEllipticIntegralSimple(double theta, double k, double* result) {
	cudaInCompleteEllipticIntegralSimple(theta, k, result);
}

void BlackHolePhysics::CompleteEllipticIntegralSimple( double k, double* result) {
	cudaCompleteEllipticIntegralSimple( k, result);
}