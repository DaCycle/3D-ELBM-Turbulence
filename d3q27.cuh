#ifndef CUDA
#define CUDA
#include <cuda_runtime.h>
#endif

#ifndef IOSSTREAM
#define IOSSTREAM
#include <iostream>
using namespace std;
#endif

#ifndef CHRONO
#define CHRONO
#include <chrono>
#endif

#ifndef MATH
#define MATH
#include <cmath>
#endif

#ifndef EQM_D3Q27_H
#define EQM_D3Q27_H
__global__ void eqm_d3q27(double* f_eq, double* Rho, double* Ux, double* Uy, double* Uz, int Cell_Count, double* w, int* Ksi);
#endif

#ifndef STREAMING_D3Q27_H
#define STREAMING_D3Q27_H
__global__ void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi);
#endif

#ifndef MOMENT_RHO_U_D3Q27_H
#define MOMENT_RHO_U_D3Q27_H
__global__ void moment_rho_u_d3q27(double* rho, double* Ux, double* Uy, double* Uz, double* f, int Cell_Count);
#endif

#ifndef COLLISION_D3Q27_H
#define COLLISION_D3Q27_H
__global__ void collision_d3q27(double* f, double* viscousity, double* f_new, double* f_eq, double Beta, int Cell_Count);
#endif