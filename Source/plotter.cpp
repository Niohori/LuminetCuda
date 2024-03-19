#include "Plotter.h"
Plotter::Plotter()
{
	//create();
}
Plotter::~Plotter() {
	// Destructor implementation
	g->disfin();
}
void Plotter::create() {

	screenWidth = GetSystemMetrics(SM_CXSCREEN);;
	screenHeight = GetSystemMetrics(SM_CYSCREEN);

	std::cout << "Screen Resolution: " << screenWidth << "x" << screenHeight << std::endl;

	g = new Dislin;
	x_max = -10000000000.0;
	x_min = 10000000000.0;
	y_max = -10000000000.0;
	y_min = 10000000000.0;
	Npoints = 0;
	g->metafl("cons");
	//g->scrmod("revers");
	g->disini();
	g->pagera();
	g->complx();
	g->nochek();
	g->unit(0);
	g->axspos(500, 0 * screenHeight - 200);
	//g->axspos(1250, 1600);
	int scrheight = std::min(screenWidth, screenHeight) - 20;
	g->axslen(scrheight, scrheight);
	//g->axslen(1200, 2800);
	g->chacod("ISO1");
	g->winfnt("Times New Roman");
	ic = g->intrgb(0., 0., 0.);
	g->axsbgd(ic);
}

void Plotter::plot_BlackHole(int Npoints, double* disk,  const double& maxX,const bool& loop) {

	double x_max = maxX;
	double x_min = 1e8;
	double y_max = -1e8;
	double y_min = 1e8;
	double factor = 1;

	double magnifier = 2.0;
	x_max /= magnifier;
	x_min = -x_max;
	y_max = x_max;
	y_min = x_min;

	/*x_max =70.0;
	x_min = -x_max;
	y_max = 2.0*3.1516;
	y_min = -0.01;*/

	if (loop) {
		g->erase();
	};
	g->setclr(0);

	g->graf(x_min, x_max, x_min, (x_max - x_min) / 10, y_min, y_max, y_min, (y_max - y_min) / 10);
	//g->setrgb(0.7, 0.7, 0.7);
	g->setrgb(0.0, 0.0, 0.0);
	g->color("fore");
	g->title();
	//g->hsymbl(5);
	//for (size_t i = 0; i < x_shad.size(); i++) {
	//	;
	//	g->setrgb(0, 1, 0);
	//	//std::cout << x_shad[i] << " , " << y_shad[i] << std::endl;
	//	g->rlsymb(21, x_shad[i], y_shad[i]);
	//}
	g->hsymbl(4);
	// Set the color map based on the color parameter
	// 
	//*******************  Secundary Image *********************************
	for (size_t i = 0; i < Npoints; i++) {
		double c = disk[i*8+7];
		if (c < 0.0)c = 0.0;
		if (c > 1.0)c = 1.0;
		g->setrgb(c, c, c);
		//g->setrgb(0, 1, 0);
		g->rlsymb(21, disk[i * 8 + 5], disk[i * 8 +6]);
	}

	//*******************  Primary Image *********************************

	for (size_t i = 0; i < Npoints; i++) {
		double c = disk[i * 8 + 4];
		if (c < 0.0)c = 0.0;
		if (c > 1.0)c = 1.0;
		g->setrgb(c, c, c);
		//g->setrgb(1, 0, 0);
		g->rlsymb(21, disk[i * 8 + 2], disk[i * 8 + 3]);
		//g->rlsymb(21, disk[i * 8 + 0], disk[i * 8 + 1]);
		//std::cout << "(x,y): (" << disk[i * 8 + 2] << ", " << disk[i * 8 + 3] << ") -> Primary Flux " << disk[i * 8 + 4]  << std::endl;
	}
	g->sendbf();
	if (loop) {
		g->endgrf();
		g->sendbf();
	};
}

void Plotter::plot_rAlpha(std::vector<double>& xx_, std::vector<double>& yy_) {
	//Dislin g;

	std::vector<double> xx = xx_;
	std::vector<double> yy =yy_;



	Npoints = xx.size();
	double x_max = -1e8;
	double x_min = 1e8;
	double y_max = -1e8;
	double y_min = 1e8;
	double factor = 1;
	for (unsigned i = 0; i < Npoints; i++)
	{
		//if (xx[i] > 40.0) { continue; }
		if (x_max < xx[i]) { x_max = xx[i]; }
		if (x_min > xx[i]) { x_min = xx[i]; }
		yy[i] *= (180 / 3.14159);
		//yy[i] =std::abs(yy[i]);
		if (y_max < yy[i]) { y_max = yy[i]; }
		if (y_min > yy[i]) { y_min = yy[i]; }
	}
	//y_min = 0.0;

	// Convert double to string with 2-digit precision
	//std::string inclination_as_string = std::to_string(inclination);
	//size_t dotPos = inclination_as_string.find('.');
	//if (dotPos != std::string::npos && dotPos + 0 < inclination_as_string.size()) {
	//	inclination_as_string = inclination_as_string.substr(0, dotPos + 0); // keep 2 digits after the dot
	//}
	//if (inclination_as_string.size() == 1) inclination_as_string = "00" + inclination_as_string;
	//if (inclination_as_string.size() == 2) inclination_as_string = "0" + inclination_as_string;
	//std::string tittel = "inclination = " + inclination_as_string + static_cast<char>(186);//static_cast<char>('\u00B0');
	//g->titlin(&tittel[0], 3);
	;	g->metafl("cons");
	//g->scrmod("revers");
	g->disini();
	g->pagera();
	g->complx();
	g->axspos(450, 1800);
	g->axslen(1200, 1200);

	g->labdig(-1, "x");
	g->ticks(9, "x");
	g->ticks(10, "y");
	ic = g->intrgb(1., 1., 1.);
	g->axsbgd(ic);
	g->graf(y_min, y_max, y_min, (y_max - y_min) / 10, x_min, x_max, x_min, (x_max - x_min) / 10);
	//g->setrgb(0.7, 0.7, 0.7);
	g->setrgb(0.0, 0.0, 0.0);
	g->color("fore");
	g->title();
	g->hsymbl(8);
	// Set the color map based on the color parameter
	// 
	//*******************  Secundary Image *********************************
	//for (size_t i = 0; i < Npoints; i++) {
	//	double c = fluxesS[i];
	//	g->setrgb(c, c, c);
	//	//g->setrgb(0, 1, 0);
	//	g->rlsymb(21, xxS[i], yyS[i]);
	//}

	//*******************  Primary Image *********************************

	for (size_t i = 0; i < Npoints; i++) {
		g->setrgb(0, 0, 0);
		g->rlsymb(21, yy[i], xx[i]);
	}

}
