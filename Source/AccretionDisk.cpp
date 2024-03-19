#include"AccretionDisk.h"

AccretionDisk::AccretionDisk(void) {
}
AccretionDisk::AccretionDisk(const double& mass, const double& incl, const double& innerradius, const double& outerradius, const int& nparticles) :
	M(mass),
	inclination(incl),
	innerRadius(innerradius),
	outerRadius(outerradius),
	nParticles(nparticles) {
	maxWidth = -1e15;
	maxFluxPrimary = -1e15;
	minFluxPrimary = 1e15;
	maxFluxSecundary = -1e15;
	minFluxSecundary = 1e15;
	plot = new Plotter;
	DIM = 0;
	plot->create();
	//std::cout << "Number of particles: " << nParticles << std::endl;
	createDisk();
};

AccretionDisk::~AccretionDisk() {
	free(Particles);//free CPU memory
	//free GPU memory

	HANDLE_ERROR(cudaFree(ParticlesCuda));
	std::cout << "GPU memory cleaned" << std::endl;
};
/**
===================================================================================================================================
* @brief The AccretionDisk class creates a disk of with nParticles scattered randomly over the inner and the outeredge
*
* @param[in] none
*
*
* @return None:
=====================================================================================================================================*/
void AccretionDisk::createDisk() {
	int nBytes = 32;
	int N = int(std::ceil(std::sqrt(nParticles) / nBytes));
	if (N < 1)N = 1;
	DIM = N * nBytes;
	nParticles = DIM * DIM;
	std::cout << " N = " << N << " DIM= " << DIM << " nParticles = " << nParticles << std::endl;
	bitmap = CPUAnimBitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	std::cout << "Bitmap environment created" << std::endl;
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap,
		bitmap.image_size()));

	std::cout << "Bitmap GPU memory allocated" << std::endl;
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_particles,
		bitmap.image_size() / 2));//contains
	std::cout << "Particles GPU memory allocated" << std::endl;
	Particles = (double*)malloc(nParticles * 8 * sizeof(double));
	std::cout << "Bitmap GPU memory allocated" << std::endl;
	// Allocate memory on CUDA device
	HANDLE_ERROR(cudaMalloc((void**)&ParticlesCuda, nParticles * 8 * sizeof(double)));
	std::cout << "Seeding particles" << std::endl;
	//ready to perform first calculations
	seedParticles(DIM, ParticlesCuda, nParticles, innerRadius, outerRadius);
	std::cout << "Creating disk...." << std::endl;
	compute(DIM, ParticlesCuda, nParticles, inclination, M);

	// Check for kernel launch errors
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copy data from device (GPU) host (CPU)
	HANDLE_ERROR(cudaMemcpy(Particles, ParticlesCuda, nParticles * 8 * sizeof(double), cudaMemcpyDeviceToHost));

	/*for (int n = 0; n < nParticles;n++) {
		double r = Particles[n * 8 + 0];
		double phi = Particles[n * 8 + 1];
		std::cout <<n<<") -> (r, phi) = (" << r << ", " << phi << ")." << std::endl;
			}*/
	std::cout << "Calculating max and min fluxes...." << std::endl;
	for (int n = 0; n < nParticles; n++) {
		if (maxWidth < Particles[n * 8 + 2]) { maxWidth = Particles[n * 8 + 2]; }
		if (maxFluxPrimary < Particles[n * 8 + 4]) { maxFluxPrimary = Particles[n * 8 + 4]; }
		if (minFluxPrimary > Particles[n * 8 + 4]) { minFluxPrimary = Particles[n * 8 + 4]; }
		if (maxFluxSecundary < Particles[n * 8 + 7]) { maxFluxSecundary = Particles[n * 8 + 7]; }
		if (minFluxSecundary > Particles[n * 8 + 7]) { minFluxSecundary = Particles[n * 8 + 7]; }
	}

	std::cout << "maxX " << maxWidth << std::endl;
	std::cout << "Primary Flux between " << minFluxPrimary << " and " << maxFluxPrimary << std::endl;
	std::cout << "Secundary Flux between " << minFluxSecundary << " and " << maxFluxSecundary << std::endl;
	for (int n = 0; n < nParticles; n++) {
		Particles[n * 8 + 4] = std::pow((std::abs(Particles[n * 8 + 4]) + minFluxPrimary) / (maxFluxPrimary + minFluxPrimary), powerScale);//Enhances "contrast"
		Particles[n * 8 + 7] = std::pow((std::abs(Particles[n * 8 + 7]) + minFluxSecundary) / (maxFluxSecundary + minFluxSecundary), powerScale) / 2;//Enhances "contrast"
		//std::cout <<"(x,y): ("<<Particles[n * 8 + 2]<< ", " << Particles[n * 8 + 3] << ") -> Primary Flux " << Particles[n * 8 + 4] << " and Secundary " << Particles[n * 8 + 7] << std::endl;
	}
	//std::cout << "And on the 0th day, God created the black hole" << std::endl;
	//
	//
	//plot->plot_BlackHole(nParticles, Particles, maxWidth, false);
	////plot.plot_rAlpha(bPrimaryParticles, alphaPrimaryParticles);
}

/**
===================================================================================================================================
* @brief play() in the AccretionDisk class increments phi and recalculates the image
*
* @param[in] none
*
*
* @return None:
=====================================================================================================================================*/
void AccretionDisk::play() {


	increment(DIM, incr, ParticlesCuda, nParticles, minFluxPrimary, maxFluxPrimary, minFluxSecundary, maxFluxSecundary, powerScale, inclination, M);

	// Check for kernel launch errors
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copy data from device (GPU) host (CPU)
	HANDLE_ERROR(cudaMemcpy(Particles, ParticlesCuda, nParticles * 8 * sizeof(double), cudaMemcpyDeviceToHost));

	//for (int n = 0; n < nParticles; n++) {
	//	Particles[n * 8 + 4] = std::pow((std::abs(Particles[n * 8 + 4]) + minFluxPrimary) / (maxFluxPrimary + minFluxPrimary), powerScale);//Enhances "contrast"
	//	Particles[n * 8 + 7] = std::pow((std::abs(Particles[n * 8 + 7]) + minFluxSecundary) / (maxFluxSecundary + minFluxSecundary), powerScale) / 2;//Enhances "contrast"
	//	//std::cout <<"(x,y): ("<<Particles[n * 8 + 2]<< ", " << Particles[n * 8 + 3] << ") -> Primary Flux " << Particles[n * 8 + 4] << " and Secundary " << Particles[n * 8 + 7] << std::endl;
	//}
	//auto t1 = std::chrono::high_resolution_clock::now();
	plot->plot_BlackHole(nParticles, Particles, maxWidth, true);
	//auto t2 = std::chrono::high_resolution_clock::now();
	//std::cout << "Time for ploting: " << (t2 - t1).count() / 1e6 << " ms." << std::endl;
}
//
void AccretionDisk::generate_frame(DataBlock* d, int tick) {
	//increment(DIM, incr, ParticlesCuda, nParticles, minFluxPrimary, maxFluxPrimary, minFluxSecundary, maxFluxSecundary, powerScale, inclination, M);

	//// Check for kernel launch errors
	//HANDLE_ERROR(cudaGetLastError());
	//HANDLE_ERROR(cudaDeviceSynchronize());

	//// Copy data from device (GPU) host (CPU)
	//HANDLE_ERROR(cudaMemcpy(Particles, ParticlesCuda, nParticles * 8 * sizeof(double), cudaMemcpyDeviceToHost));

	//for (int n = 0; n < nParticles; n++) {
	//	Particles[n * 8 + 4] = std::pow((std::abs(Particles[n * 8 + 4]) + minFluxPrimary) / (maxFluxPrimary + minFluxPrimary), powerScale);//Enhances "contrast"
	//	Particles[n * 8 + 7] = std::pow((std::abs(Particles[n * 8 + 7]) + minFluxSecundary) / (maxFluxSecundary + minFluxSecundary), powerScale) / 2;//Enhances "contrast"
	//	//std::cout <<"(x,y): ("<<Particles[n * 8 + 2]<< ", " << Particles[n * 8 + 3] << ") -> Primary Flux " << Particles[n * 8 + 4] << " and Secundary " << Particles[n * 8 + 7] << std::endl;
	//}
	//auto t1 = std::chrono::high_resolution_clock::now();
	//plot->plot_BlackHole(nParticles, Particles, maxWidth, true);
	//auto t2 = std::chrono::high_resolution_clock::now();
	//std::cout << "Time for ploting: " << (t2 - t1).count() / 1e6 << " ms." << std::endl;
}