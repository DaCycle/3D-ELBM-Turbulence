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
__global__ void eqm_d3q27(double* f_eq, double* rho, double* U, int N_x, int N_y);
#endif

#ifndef STREAMING_D3Q27_H
#define STREAMING_D3Q27_H
__global__ void streaming_d3q27(double* f_new, double* f, double* f_eq_BC, double* Rho, double U_lid, double A_TL, double A_TR, double A_TM, int N_x, int N_y);
#endif

#ifndef MOMENT_RHO_U_D3Q27_H
#define MOMENT_RHO_U_D3Q27_H
__global__ void moment_rho_u_d3q27(double* rho, double* U, double* f, int N_x, int N_y);
#endif

#ifndef COLLISION_D3Q27_H
#define COLLISION_D3Q27_H
__global__ void collision_d3q27(double* f, double* viscousity, double* f_new, double* f_eq, double Beta, int N_x, int N_y);
#endif

#ifndef CURL_2D_H
#define CURL_2D_H
__global__ void curl_2D(double* curl, double* U, int N_x, int N_y);
#endif

#pragma once
extern __constant__ double w[9];
extern __constant__ double Ksi[9][2];

// Replace with template is faster

//#pragma once
//template<int Q>
//struct LBMConstants;
//
//template<>
//struct LBMConstants<9> {
//    static __device__ __host__ constexpr double w[9] = {
//        4.0 / 9.0,
//        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
//        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
//    };
//
//    static __device__ __host__ constexpr double Ksi[9][2] = {
//        { 0.0, 0.0 },
//        { 1.0, 0.0 },
//        { 0.0, 1.0 },
//        { -1.0, 0.0 },
//        { 0.0, -1.0 },
//        { 1.0, 1.0 },
//        { -1.0, 1.0 },
//        { -1.0, -1.0 },
//        { 1.0, -1.0 }
//    };
//};