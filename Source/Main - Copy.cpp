#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <map>
#include <cstdlib>   // For rand() function
#include <ctime>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <cuda_runtime.h>
#include <cudastuff.h>
double snTaylor(double k, double u) {
	double sn = u - (1 + k * k) * std::pow(u, 3) / 6.0 + (1 + 14.0 * k * k + k * k * k * k) * std::pow(u, 4) / (5 * 4 * 3 * 2)
		- (1 + 135.0 * k * k + 135.0 * k * k * k * k + k * k * k * k * k * k) * std::pow(u, 7) / (7 * 6 * 5 * 4 * 3 * 2);
	return sn;
}

// Function to generate a random number between 0 and 1 (inclusive)
double randomZeroToOne() {
	// Seed the random number generator
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	// Generate a random integer between 0 and RAND_MAX
	int randomInt = std::rand();

	// Scale the random integer to fit within [0, 1]
	double randomZeroToOne = static_cast<double>(randomInt) / RAND_MAX;

	return randomZeroToOne;
}
const double M_PI = 3.14159265358979323846;
int main() {
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> disk(0.0, 1.0);
	std::uniform_real_distribution<> distheta(0.0, M_PI / 2 * 0.99);
	int N = 10000;
	double* d_result_3, h_result_3;

	cudaMalloc((void**)&d_result_3, sizeof(double));
	printf("Incomplete Elliptic integral of the first kind F()\n");
	for (int n = 0; n < N; n++) {
		double k = disk(gen);
		//k = randomZeroToOne();
		//k = 1.0;
		//k = std::sin(int(std::asin(k) * 180 / M_PI) * M_PI / 180);
		double theta = distheta(gen);
		//theta = randomZeroToOne() * M_PI / 2;
		//theta = 0.0*M_PI / 2;
		//theta = int(theta * 180 / M_PI) * M_PI / 180;
		double ell_inf_boost = std::ellint_1(k, theta);  //incomplete elliptic integral
		if (theta == M_PI / 2) {
			BlackHolePhysics::CompleteEllipticIntegralSimple(k, d_result_3);
			//BlackHolePhysics::CompleteEllipticIntegralK(k, d_result_3);
			//BlackHolePhysics::InCompleteEllipticIntegralFukushima(theta, k, d_result_3);
		}
		else {
			BlackHolePhysics::InCompleteEllipticIntegralSimple(theta, k, d_result_3);
			//BlackHolePhysics::InCompleteEllipticIntegralFukushima(theta, k, d_result_3);
		}
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		if (std::abs(ell_inf_boost - h_result_3) > 0.01) {
			printf("Boost: F(%f,%f) = %f", k, theta * 180 / M_PI, ell_inf_boost);
			printf(" <---> Cuda: F(%f,%f) = %f", k, theta * 180 / M_PI, h_result_3);
			printf(" <---> Delta = %f\n", ell_inf_boost - h_result_3);
		}
	}
	std::cout << std::endl << "******************************************************************" << std::endl;
	printf("Jacobi Elliptic function sn()\n");
	for (int n = 0; n < N; n++) {
		double k = disk(gen);
		//k = 1.0;
		//k = randomZeroToOne();
		//k = std::sin(int(std::asin(k) * 180 / M_PI) * M_PI / 180);
		double theta = distheta(gen);
		//theta = randomZeroToOne() * M_PI / 2;
		//theta = M_PI / 2;
		//theta = int(theta * 180 / M_PI) * M_PI / 180;
		double ell_inf_boost = std::ellint_1(k, theta);  //incomplete elliptic integral
		if (theta == M_PI / 2) {
			BlackHolePhysics::CompleteEllipticIntegralK(k, d_result_3);
			//BlackHolePhysics::InCompleteEllipticIntegralFukushima(theta, k, d_result_3);
		}
		else {
			BlackHolePhysics::InCompleteEllipticIntegralFukushima(theta, k, d_result_3);
		}
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		double snBoost = boost::math::jacobi_sn(k, ell_inf_boost); //Boost Jacobi elliptic function sn
		BlackHolePhysics::JacobiSnSimple(k, h_result_3, d_result_3); //Cuda Jacobi elliptic function sn
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		// h_result_3 = snTaylor(k, ell_inf_boost);
		if (std::abs(snBoost - h_result_3) > 0.01) {
			printf("Boost: sn(%f) = %f", k, std::asin(snBoost) * 180 / M_PI);
			printf(" <---> Cuda: sn(%f) = %f", k, std::asin(h_result_3) * 180 / M_PI);
			printf(" <---> Delta = %f\n", (std::asin(snBoost) - std::asin(h_result_3)) * 180 / M_PI);
		}
	}

	int nSteps = 100000;
	double step = 1.0 / nSteps;
	double theta = M_PI / 2.1;
	std::vector<double> args;
	std::cout << "Phi = " << theta * 180 / M_PI << std::endl;
	std::cout << std::endl << "======================================= Integrals =============================================================================" << std::endl << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();
	for (double kk = 0.1; kk < 1.0; kk = kk + step) {
		BlackHolePhysics::InCompleteEllipticIntegralFukushima(theta, std::sqrt(kk * kk), d_result_3);
		//BlackHolePhysics::CompleteEllipticIntegralK(kk, d_result_3);
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		args.push_back(h_result_3);
	}
	args.clear();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Time (s) for Cuda Fukushima method " << (t2 - t1).count() / 1e3/nSteps << " micros/integral." << std::endl;
	std::cout << std::endl << "=====================================================================================================================================================" << std::endl << std::endl;

	std::cout << "Other method" << std::endl;
	
	t1 = std::chrono::high_resolution_clock::now();
	for (double kk = 0.1; kk < 1.0; kk = kk +step) {
		BlackHolePhysics::InCompleteEllipticIntegralSimple(theta, std::sqrt(kk * kk), d_result_3);
		//BlackHolePhysics::CompleteEllipticIntegralK(kk, d_result_3);
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		args.push_back(h_result_3);
		//printf("Incomplete Elliptic integral of the first kind F(%f) = %f\n", kk, h_result_3);
	}
	t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Time (s) for Cuda simple method " << (t2 - t1).count() / 1e3 / nSteps << " micros/integral." << std::endl;
	std::cout << std::endl << "=====================================================================================================================================================" << std::endl << std::endl;
	std::cout << std::endl << "=====================================================================================================================================================" << std::endl << std::endl;

	std::cout << "======================= BOOST Jacobi sn()  -------------------------" << std::endl;
	int count = 0;
	t1 = std::chrono::high_resolution_clock::now();
	for (double kk = 0.1; kk < 1.0; kk = kk + step) {
		double arg = args[count];
		count++;
		BlackHolePhysics::JacobiSN(std::sqrt(kk * kk), arg, d_result_3); //Cuda Jacobi elliptic function sn
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
	/*	printf("Jacobi sn(%f) = %f ", kk, sn);
		printf("    (phi  = %f)", std::asin(sn) * 180 / M_PI);
		printf("    for arg   = %f)\n", arg);*/
	}
	t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Time (s) for Cuda first method " << (t2 - t1).count() / 1e3 / nSteps << " micros/integral." << std::endl;
	std::cout << std::endl << "=====================================================================================================================================================" << std::endl << std::endl;
	std::cout << "======================= CUDAJacobi sn()  -------------------------" << std::endl;
	count = 0;
	t1 = std::chrono::high_resolution_clock::now();
	for (double kk = 0.1; kk < 1.0; kk = kk + step) {
		double arg = args[count];
		count++;
		BlackHolePhysics::JacobiSnSimple(std::sqrt(kk * kk), arg, d_result_3); //Cuda Jacobi elliptic function sn
		cudaMemcpy(&h_result_3, d_result_3, sizeof(double), cudaMemcpyDeviceToHost);
		//printf("Jacobi sn(%f) = %f ", kk, h_result_3);
		//printf("    (phi  = %f)", std::asin(h_result_3) * 180 / M_PI);
		//printf("    for arg   = %f)\n", arg);
	}
	t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Time (s) for Cuda second method " << (t2 - t1).count() / 1e3 / nSteps << " micros/integral." << std::endl;
	cudaFree(d_result_3);
	return 0;
}