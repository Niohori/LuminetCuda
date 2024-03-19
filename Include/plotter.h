#pragma once
#ifndef PLOTTER_H
#define PLOTTER_H
//#include "winuser.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <sstream>
#include <array>
#include <exception>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <map>


#include <discpp.h>
#include <Windows.h>
// Undefine macros that may cause conflicts
#undef min
#undef max
const double a_PI = 3.14159265358979323846;

class Plotter {
public:
	Plotter();
	~Plotter();
	void create();
void plot_BlackHole(int,  double*, const double&,const bool&);
void plot_rAlpha( std::vector<double>& xx_, std::vector<double>&);

private:
private://variables
	int screenWidth;
	int screenHeight;
	Dislin* g;
	int ic;//dislin
	double x_max;
	double x_min;
	double y_max;
	double y_min;
	int Npoints;
	// Vectors to store RGB color values
	std::vector<std::tuple<double, double, double>> rgbVector;
	std::vector<std::tuple<double, double, double>> rgbVector_g;
};
#endif