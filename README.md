# LuminetCuda
Cuda version of C++ version Luminet project.
Try to animate BH disk.
Messy code.
# Partial results :
Implemented Elliptic Integrals and function on GPU (both Toshio Fukushima algorithm as Carlson's ).

Moved to bmp rendering with glut instead of Dislin due to bad performance for image rendering.

Using reverse raytracing to speed up calculations.
<img src="https://github.com/Niohori/LuminetCuda/blob/main/Documentation/AnimatedBlackHole.gif" width="800" />
<img src="https://github.com/Niohori/LuminetCuda/blob/main/Documentation/BoringBlackHole.PNG" width="800" />

# TODO
Populate with realistic accretion disk (Novikov & Thorne (1973) and Page & Thorne (1974))?.
## Bugs
Reverse raytracing causes interference patterns due to rounding errors. How to improve?

## Improvements
Clean code.
# Dependencies
- CUDA
- Glut
