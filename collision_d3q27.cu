#include "d3q27.cuh"

__global__
void collision_d3q27(double* f, double* viscousity, double* f_new, double* f_eq, double Beta, int N_x, int N_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= N_x * N_y) return; // Ensure we don't access out of bounds

	for (int i = idx; i < N_x * N_y; i += stride) {
		// Initialize Variables
		double z[9];
		double z_neg[9] = { 0.0 };
		double z_max = 0;
		double z_min = INFINITY;
		double f_neg[9] = { 0.0 };
		double a = 0;
		double b = 0;
		double c = 0;
		double alpha;
		int index;

		// Calculate z and z_neg values
		for (int d = 0; d < 9; d++) {
			index = i * 9 + d;
			z[d] = f_eq[index] / f_new[index] - 1;
			if (z[d] < 0) {
				z_neg[d] = z[d];
				f_neg[d] = f_new[index];
			}
			z_max = max(z_max, z[d]);
			z_min = min(z_min, z[d]);
		}

		// Calculate a, b, and c for quadratic equation
		for (int d = 0; d < 9; d++) {
			index = i * 9 + d;
			a += f_neg[d] * pow(z_neg[d], 3) / 2;
			b += f_new[index] * pow(z[d], 2) / 2;
			c += f_new[index] * 2 * pow(z[d], 2) / (2 + z[d]);
		}

		// Calculate alpha
		double denominator = b + sqrt(pow(b, 2) - 4 * a * c);
		if (denominator == 0) {
			alpha = 2;
		}
		else {
			alpha = 2 * c / denominator;
		}

		//// Calculate alpha max
		//double alpha_max = -1 / (Beta * z_min);
		//if (alpha > alpha_max) {
		//	alpha = (1.0 + alpha_max) / 2.0;
		//}

		// Collision Step
		for (int d = 0; d < 9; d++) {
			index = i * 9 + d;
			f[index] = f_new[index] + alpha * Beta * (f_eq[index] - f_new[index]);
		}

		// Write Viscousity
		//viscousity[i] = (1.0 / alpha - 0.5) / 3.0;
		viscousity[i] = (1.0 / alpha / Beta - 0.5) / 3.0;
	}
}