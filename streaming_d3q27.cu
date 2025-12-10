#include "d3q27.cuh"

__global__
void streaming_d3q27(double* f_new, double* f, double* f_eq_BC, double* Rho, double U_lid, double A_TL, double A_TR, double A_TM, int N_x, int N_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= N_x * N_y) return; // Ensure we don't access out of bounds

	for (int index = idx; index < N_x * N_y; index += stride) {
		int i = index % N_x;
		int j = index / N_x;
		int x_max = N_x - 1;
		int y_max = N_y - 1;

		double Rho_t; // Temporary density variable
		double A; // Constant for Maxwell Diffuse Boundary Conditions

		// Instructions for navigating the grid
		// f(d, j, i)
		// i + 1 -> index + 1
		// i - 1 -> index - 1
		// j + 1 -> index + N_x
		// j - 1 -> index - N_x
		// d     -> (index +- __) * 9 + d

		if (j == 0) { // Top Boundary
			if (i == 0) { // Top Left
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
				f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
				f_new[index * 9 + 6] = f[(index + N_x + 1) * 9 + 6];
				// Unkown PDFS
				Rho_t = (f_new[index * 9 + 0] + f_new[index * 9 + 2] + f_new[index * 9 + 3] + f_new[index * 9 + 6]) / (1 - A_TL);
				f_new[index * 9 + 1] = Rho_t / 9.0;
				f_new[index * 9 + 4] = Rho_t * (1 - 1.5 * U_lid * U_lid) / 9.0;
				f_new[index * 9 + 5] = Rho_t / 36.0;
				f_new[index * 9 + 7] = Rho_t * (1 - 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
				f_new[index * 9 + 8] = Rho_t * (1 + 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
			}
			else if (i == x_max) { // Top Right
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
				f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
				f_new[index * 9 + 5] = f[(index + N_x - 1) * 9 + 5];
				// Unkown PDFS
				Rho_t = (f_new[index * 9 + 0] + f_new[index * 9 + 1] + f_new[index * 9 + 2] + f_new[index * 9 + 5]) / (1 - A_TR);
				f_new[index * 9 + 3] = Rho_t / 9.0;
				f_new[index * 9 + 4] = Rho_t * (1 - 1.5 * U_lid * U_lid) / 9.0;
				f_new[index * 9 + 6] = Rho_t / 36.0;
				f_new[index * 9 + 7] = Rho_t * (1 - 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
				f_new[index * 9 + 8] = Rho_t * (1 + 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
			}
			else { // Top Middle
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
				f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
				f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
				f_new[index * 9 + 5] = f[(index + N_x - 1) * 9 + 5];
				f_new[index * 9 + 6] = f[(index + N_x + 1) * 9 + 6];
				// Unkown PDFS
				Rho_t = (f_new[index * 9 + 0] + f_new[index * 9 + 1] + f_new[index * 9 + 2] + f_new[index * 9 + 3] + f_new[index * 9 + 5] + f_new[index * 9 + 6]) / (1 - A_TM);
				f_new[index * 9 + 4] = Rho_t * (1 - 1.5 * U_lid * U_lid) / 9.0;
				f_new[index * 9 + 7] = Rho_t * (1 - 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
				f_new[index * 9 + 8] = Rho_t * (1 + 3.0 * U_lid + 3.0 * U_lid * U_lid) / 36.0;
			}
		}
		else if (j == y_max) { // Bottom Boundary
			if (i == 0) { // Bottom Left
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
				f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
				f_new[index * 9 + 7] = f[(index - N_x + 1) * 9 + 7];
				// Unkown PDFS
				Rho_t = 36.0 * (f_new[index * 9 + 0] + f_new[index * 9 + 3] + f_new[index * 9 + 4] + f_new[index * 9 + 7]) / 25.0;
				f_new[index * 9 + 1] = Rho_t / 9.0;
				f_new[index * 9 + 2] = Rho_t / 9.0;
				f_new[index * 9 + 5] = Rho_t / 36.0;
				f_new[index * 9 + 6] = Rho_t / 36.0;
				f_new[index * 9 + 8] = Rho_t / 36.0;
			}
			else if (i == x_max) { // Bottom Right
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
				f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
				f_new[index * 9 + 8] = f[(index - N_x - 1) * 9 + 8];
				// Unkown PDFS
				Rho_t = 36.0 * (f_new[index * 9 + 0] + f_new[index * 9 + 1] + f_new[index * 9 + 4] + f_new[index * 9 + 8]) / 25.0;
				f_new[index * 9 + 2] = Rho_t / 9.0;
				f_new[index * 9 + 3] = Rho_t / 9.0;
				f_new[index * 9 + 6] = Rho_t / 36.0;
				f_new[index * 9 + 5] = Rho_t / 36.0;
				f_new[index * 9 + 7] = Rho_t / 36.0;
			}
			else { // Bottom Middle
				f_new[index * 9 + 0] = f[index * 9 + 0];
				f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
				f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
				f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
				f_new[index * 9 + 7] = f[(index - N_x + 1) * 9 + 7];
				f_new[index * 9 + 8] = f[(index - N_x - 1) * 9 + 8];
				// Unkown PDFS
				Rho_t = 6.0 * (f_new[index * 9 + 0] + f_new[index * 9 + 1] + f_new[index * 9 + 3] + f_new[index * 9 + 4] + f_new[index * 9 + 7] + f_new[index * 9 + 8]) / 5.0;
				f_new[index * 9 + 2] = Rho_t / 9.0;
				f_new[index * 9 + 5] = Rho_t / 36.0;
				f_new[index * 9 + 6] = Rho_t / 36.0;
			}
		}
		else if (i == 0) { // Left Boundary
			f_new[index * 9 + 0] = f[index * 9 + 0];
			f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
			f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
			f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
			f_new[index * 9 + 6] = f[(index + N_x + 1) * 9 + 6];
			f_new[index * 9 + 7] = f[(index - N_x + 1) * 9 + 7];
			// Unkown PDFS
			Rho_t = 6.0 * (f_new[index * 9 + 0] + f_new[index * 9 + 2] + f_new[index * 9 + 3] + f_new[index * 9 + 4] + f_new[index * 9 + 6] + f_new[index * 9 + 7]) / 5.0;
			f_new[index * 9 + 1] = Rho_t / 9.0;
			f_new[index * 9 + 5] = Rho_t / 36.0;
			f_new[index * 9 + 8] = Rho_t / 36.0;
		}
		else if (i == x_max) { // Right Boundary
			f_new[index * 9 + 0] = f[index * 9 + 0];
			f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
			f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
			f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
			f_new[index * 9 + 5] = f[(index + N_x - 1) * 9 + 5];
			f_new[index * 9 + 8] = f[(index - N_x - 1) * 9 + 8];
			// Unkown PDFS
			Rho_t = 6.0 * (f_new[index * 9 + 0] + f_new[index * 9 + 2] + f_new[index * 9 + 3] + f_new[index * 9 + 4] + f_new[index * 9 + 6] + f_new[index * 9 + 7]) / 5.0;
			f_new[index * 9 + 3] = Rho_t / 9.0;
			f_new[index * 9 + 6] = Rho_t / 36.0;
			f_new[index * 9 + 7] = Rho_t / 36.0;
		}
		else { // Interior Nodes
			f_new[index * 9 + 0] = f[index * 9 + 0];
			f_new[index * 9 + 1] = f[(index - 1) * 9 + 1];
			f_new[index * 9 + 2] = f[(index + N_x) * 9 + 2];
			f_new[index * 9 + 3] = f[(index + 1) * 9 + 3];
			f_new[index * 9 + 4] = f[(index - N_x) * 9 + 4];
			f_new[index * 9 + 5] = f[(index + N_x - 1) * 9 + 5];
			f_new[index * 9 + 6] = f[(index + N_x + 1) * 9 + 6];
			f_new[index * 9 + 7] = f[(index - N_x + 1) * 9 + 7];
			f_new[index * 9 + 8] = f[(index - N_x - 1) * 9 + 8];
		}
	}
}