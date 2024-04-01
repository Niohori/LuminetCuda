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
	DIM = 0;
	//plot->create();
	//std::cout << "Number of particles: " << nParticles << std::endl;
	//createDisk();
	createDiskBMP();
};

AccretionDisk::~AccretionDisk() {
	//free(Particles);//free CPU memory
	////free GPU memory

	//HANDLE_ERROR(cudaFree(ParticlesCuda));
	//std::cout << "GPU memory cleaned" << std::endl;
};




void AccretionDisk::createDiskBMP() {

	bitmap = CPUAnimBitmap(BMPDIM, BMPDIM, &data);
	data.bitmap = &bitmap;
	std::cout << "Bitmap environment created with  " << bitmap.image_size()/ BMPDIM/BMPDIM << " elements." << std::endl;
	
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap,
		bitmap.image_size()));//bitmap.image_size=4*BMPDIM*BMPDIM,
	std::cout << "Bitmap GPU memory allocated" << std::endl;

	HANDLE_ERROR(cudaMalloc((void**)&data.dev_box_xy, 11 * BMPDIM * BMPDIM * sizeof(double)));//8
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_BHDisk,1 * BMPDIM * BMPDIM * sizeof(double)));
	std::cout << "Particles GPU memory allocated" << std::endl;
	makeDisk(&data, inclination,  M,outerRadius);

}

void AccretionDisk::playBMP(double bh_mass, double outerRadius) {
	std::cout << "Press ESC to stop animation ." << std::endl;
	//HANDLE_ERROR(cudaMemcpy(data.bitmap->get_ptr(),
	//	data.dev_bitmap,
	//	data.bitmap->image_size(),
	//	cudaMemcpyDeviceToHost));
	bitmap.anim_and_exit((void (*)(void*, int))generate_frame,(void (*)(void*))cleanup);
	return;
}