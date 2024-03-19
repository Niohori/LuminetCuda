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
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//#include "book.h"
#include "cpu_anim.h"
#include <BlackHolePhysics.h>
#include "AccretionDisk.h"


 //clean up memory allocated on the GPU
//void cleanup(DataBlock* d) {
//	HANDLE_ERROR(cudaFree(d->dev_bitmap));
//}

int main() {
	const double inclination = 80.0;

	const double bh_mass = 1.;// provide black hole mass value
	const double radius = 70.0 * bh_mass;//  provide radius value > 6M

	AccretionDisk aDisk(bh_mass, inclination * M_PI / 180, 6 * bh_mass, radius, 10000);//creates disk and makes first calculations
	while (true) {
		aDisk.play();
	}
	//size_t frame = 1;
	//auto t1 = std::chrono::high_resolution_clock::now();
	//auto t2 = std::chrono::high_resolution_clock::now();
	//while (true) {
	//	t1 = std::chrono::high_resolution_clock::now();
	//	aDisk.play();
	//	t2 = std::chrono::high_resolution_clock::now();
	//	frame++;
	//	std::cout << "Frame " << frame << " time:" << (t2 - t1).count() / 1e6 << " ms." << std::endl;
	//}
	//DataBlock   data;
	//CPUAnimBitmap  bitmap(BMPDIM, BMPDIM, &data);
	//data.bitmap = &bitmap;

	//HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap,
	//	bitmap.image_size()));
	//
	//// Define lambda functions for generate_frame and cleanup
	//auto generate_frame_lambda = [&](void* d, int i) {
	//	aDisk.generate_frame(static_cast<DataBlock*>(d), i);
	//	};

	//auto cleanup_lambda = [&](void* d) {
	//	cleanup(static_cast<DataBlock*>(d));
	//	};

	//// Call anim_and_exit with the lambda functions
	//bitmap.anim_and_exit(generate_frame_lambda, cleanup_lambda);
	return 0;
}