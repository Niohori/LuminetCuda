#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <cmath>
#include <functional>
#include <map>
#include <stdlib.h> // Include stdlib.h for EXIT_FAILURE

#define PII  3.14159265359

//clean up memory allocated on the GPU
//void cleanup(DataBlock* d) {
//	HANDLE_ERROR(cudaFree(d->dev_bitmap));
//}
int  CubicRoot(double b, double M, double* symRoots, double& root) {
	//Viète formulae
	double p = -b * b;
	double q = 2.0 * b * b*M;
	std::vector<double> roots(3);
	roots[0]=symRoots[0] = 2.0f * std::sqrt(-p / 3.0f) * std::cos(1.0f / 3.0f * std::acos(3.0f / 2.0f * q / p * std::sqrt(-3.0f / p)) - 2.0f / 3.0f * PII * 0);
	roots[1] = symRoots[1] = 2.0f * std::sqrt(-p / 3.0f) * std::cos(1.0f / 3.0f * std::acos(3.0f / 2.0f * q / p * std::sqrt(-3.0f / p)) - 2.0f / 3.0f * PII * 1);
	roots[1] = symRoots[2] = 2.0f * std::sqrt(-p / 3.0f) * std::cos(1.0f / 3.0f * std::acos(3.0f / 2.0f * q / p * std::sqrt(-3.0f / p)) - 2.0f / 3.0f * PII * 2);
	root = *std::max_element(roots.begin(), roots.end());
	return 3;
}
int main() {
	double roots[3];
	double symRoots[3];
	double root;
	double step = 1.0;
	double M = 1;
	
	for (double b = 3.0f * std::sqrt(3.0f) + step / 10.0; b < 70.0; b += step)
	{

		//std::cout << "Viéte condition for real roots : " << 4.0f * std::pow(-b * b, 3) + 27.0f * std::pow(2.0 * b * b, 2) <<  std::endl;
		int n = CubicRoot(b, M, symRoots, root);

		std::cout << n << " roots for b = " << b << ": ";

		for (int i = 0; i < n; i++) {
			//std::cout << roots[i] << " -> f=   "<<std::pow(roots[i],3)  -b * b* roots[i]+2.0*b * b <<",    ";
			std::cout << symRoots[i] << ", ";
		}
		std::cout << " return root = " << root << std::endl;
	}
	return 0;
}