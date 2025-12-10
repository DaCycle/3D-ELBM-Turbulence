#include "d3q27.cuh"

__global__
void moment_rho_u_d3q27(double* rho, double* U, double* f, int N_x, int N_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= N_x * N_y) return; // Ensure we don't access out of bounds

	// Calculate density and velocity
	for (int i = idx; i < N_x * N_y; i += stride)
	{
		rho[i] = f[i * 9] + f[i * 9 + 1] + f[i * 9 + 2] + f[i * 9 + 3] + f[i * 9 + 4] + f[i * 9 + 5] + f[i * 9 + 6] + f[i * 9 + 7] + f[i * 9 + 8];
		U[i] = (f[i * 9 + 1] - f[i * 9 + 3] + f[i * 9 + 5] - f[i * 9 + 6] - f[i * 9 + 7] + f[i * 9 + 8]) / rho[i]; // x
		U[i + N_x * N_y] = (f[i * 9 + 2] - f[i * 9 + 4] + f[i * 9 + 5] + f[i * 9 + 6] - f[i * 9 + 7] - f[i * 9 + 8]) / rho[i]; // y
	}
}