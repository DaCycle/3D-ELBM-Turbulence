#include "d3q27.cuh"

__global__
void moment_rho_u_d3q27(double* rho, double* Ux, double* Uy, double* Uz, double* f, int Cell_Count) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds

	// Calculate density and velocity
	for (int i = idx; i < Cell_Count; i += stride)
	{
		rho[i] = f[i* 27] + f[i * 27 + 1] + f[i * 27 + 2] + f[i * 27 + 3] + f[i * 27 + 4] + f[i * 27 + 5] + f[i * 27 + 6] + f[i * 27 + 7] + f[i * 27 + 8]
			+ f[i * 27 + 9] + f[i * 27 + 10] + f[i * 27 + 11] + f[i * 27 + 12] + f[i * 27 + 13] + f[i * 27 + 14]
			+ f[i * 27 + 15] + f[i * 27 + 16] + f[i * 27 + 17] + f[i * 27 + 18] + f[i * 27 + 19] + f[i * 27 + 20]
			+ f[i * 27 + 21] + f[i * 27 + 22] + f[i * 27 + 23] + f[i * 27 + 24] + f[i * 27 + 25] + f[i * 27 + 26];
		Ux[i] = (f[i * 27 + 1] - f[i * 27 + 2] + f[i * 27 + 7] - f[i * 27 + 8] + f[i * 27 + 9] - f[i * 27 + 10]
			+ f[i * 27 + 11] - f[i * 27 + 12] + f[i * 27 + 13] - f[i * 27 + 14] + f[i * 27 + 19] - f[i * 27 + 20]
			+ f[i * 27 + 21] - f[i * 27 + 22] + f[i * 27 + 23] - f[i * 27 + 24] + f[i * 27 + 25] - f[i * 27 + 26]) / rho[i]; // x
		Uy[i] = (f[i * 27 + 3] - f[i * 27 + 4] + f[i * 27 + 7] - f[i * 27 + 8] - f[i * 27 + 9] + f[i * 27 + 10]
			+ f[i * 27 + 15] - f[i * 27 + 16] + f[i * 27 + 17] - f[i * 27 + 18] + f[i * 27 + 19] - f[i * 27 + 20]
			+ f[i * 27 + 21] - f[i * 27 + 22] - f[i * 27 + 23] + f[i * 27 + 24] - f[i * 27 + 25] + f[i * 27 + 26]) / rho[i]; // y
		Uz[i] = (f[i * 27 + 5] - f[i * 27 + 6] + f[i * 27 + 11] - f[i * 27 + 12] - f[i * 27 + 13] + f[i * 27 + 14]
			+ f[i * 27 + 15] - f[i * 27 + 16] - f[i * 27 + 17] + f[i * 27 + 18] + f[i * 27 + 19] - f[i * 27 + 20]
			- f[i * 27 + 21] + f[i * 27 + 22] + f[i * 27 + 23] - f[i * 27 + 24] - f[i * 27 + 25] + f[i * 27 + 26]) / rho[i]; // z
	}
}

