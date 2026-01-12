#include "d3q27.cuh"

__global__
void eqm_d3q27(double* f_eq, double* Rho, double* Ux, double* Uy, double* Uz, int Cell_Count, double* w, int* Ksi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds

	for (int i = idx; i < Cell_Count; i += stride) {

		// Entropic Lattice Boltzmann method equilibrium distribution function
		double UU = Ux[i] * Ux[i] + Uy[i] * Uy[i] + Uz[i] * Uz[i];

		// Initial guess for lagrangian multipliers and define Residuals
		double alpha = std::log(Rho[i]) - 1.5 * UU;
		double beta_x = 3.0 * Ux[i];
		double beta_y = 3.0 * Uy[i];
		double beta_z = 3.0 * Uz[i];

		// Initialize variables for Newton's method
		double R[4] = { 1.0, 1.0, 1.0, 1.0 };
		double J[4][4] = { 0.0 };
		double L[4][4] = { 0.0 };
		double y[4] = { 0.0 };
		double delta[4] = { 0.0 };
		//double R_0 = 1.0;
		//double R_x = R_0;
		//double R_y = R_0;
		//double J_11;
		//double J_12;
		//double J_13;
		//double J_22;
		//double J_23;
		//double J_33;
		//double L_11;
		//double L_12;
		//double L_13;
		//double L_22;
		//double L_23;
		//double L_33;
		//double y1;
		//double y2;
		//double y3;
		//double delta_alpha;
		//double delta_beta_x;
		//double delta_beta_y;

		// Newton's method loop
		for (int j = 0; j < 100; j++) {
			// Calculate this iteration's f_eq
			for (int d = 0; d < 27; d++) {
				f_eq[i * 27 + d] = w[d] * exp(alpha + beta_x * Ksi[3 * d] + beta_y * Ksi[3 * d + 1] + beta_z * Ksi[3 * d + 2]);
			}

			// Calculate Residuals
			#pragma unroll
			for (int d = 0; d < 27; d++) {
				J[0][0] += f_eq[i * 27 + d];
				J[0][1] += f_eq[i * 27 + d] * Ksi[3 * d];
				J[0][2] += f_eq[i * 27 + d] * Ksi[3 * d + 1];
				J[0][3] += f_eq[i * 27 + d] * Ksi[3 * d + 2];
			}
			R[0] = J[0][0] - Rho[i];
			R[1] = J[0][1] - Ux[i] * Rho[i];
			R[2] = J[0][2] - Uy[i] * Rho[i];
			R[3] = J[0][3] - Uz[i] * Rho[i];

			if (fabs(R[0]) < 1e-4 && fabs(R[1]) < 1e-4 && fabs(R[2]) < 1e-4 && fabs(R[3]) < 1e-4) {
				break;
			}

			// Calculate rest of Jacobian
			#pragma unroll
			for (int d = 0; d < 27; d++) {
				J[1][1] += f_eq[i * 27 + d] * Ksi[3 * d] * Ksi[3 * d];
				J[1][2] += f_eq[i * 27 + d] * Ksi[3 * d] * Ksi[3 * d + 1];
				J[1][3] += f_eq[i * 27 + d] * Ksi[3 * d] * Ksi[3 * d + 2];
				J[2][2] += f_eq[i * 27 + d] * Ksi[3 * d + 1] * Ksi[3 * d + 1];
				J[2][3] += f_eq[i * 27 + d] * Ksi[3 * d + 1] * Ksi[3 * d + 2];
				J[3][3] += f_eq[i * 27 + d] * Ksi[3 * d + 2] * Ksi[3 * d + 2];
			}

			// Cholesky decomposition to avoid inverse of Jacobian
			L[0][0] = sqrt(J[0][0]);
			L[0][1] = J[0][1] / L[0][0];
			L[0][2] = J[0][2] / L[0][0];
			L[0][3] = J[0][3] / L[0][0];
			L[1][1] = sqrt(J[1][1] - L[0][1] * L[0][1]);
			L[1][2] = (J[1][2] - L[0][1] * L[0][2]) / L[1][1];
			L[1][3] = (J[1][3] - L[0][1] * L[0][3]) / L[1][1];
			L[2][2] = sqrt(J[2][2] - L[0][2] * L[0][2] - L[1][2] * L[1][2]);
			L[2][3] = (J[2][3] - L[0][2] * L[0][3] - L[1][2] * L[1][3]) / L[2][2];
			L[3][3] = sqrt(J[3][3] - L[0][3] * L[0][3] - L[1][3] * L[1][3] - L[2][3] * L[2][3]);

			y[0] = R[0] / L[0][0];
			y[1] = (R[1] - L[0][1] * y[0]) / L[1][1];
			y[2] = (R[2] - L[0][2] * y[0] - L[1][2] * y[1]) / L[2][2];
			y[3] = (R[3] - L[0][3] * y[0] - L[1][3] * y[1] - L[2][3] * y[2]) / L[3][3];

			delta[3] = y[3] / L[3][3];
			delta[2] = (y[2] - L[2][3] * delta[3]) / L[2][2];
			delta[1] = (y[1] - L[1][2] * delta[2] - L[1][3] * delta[3]) / L[1][1];
			delta[0] = (y[0] - L[0][1] * delta[1] - L[0][2] * delta[2] - L[0][3] * delta[3]) / L[0][0];

			// Update lagrangian multipliers
			alpha -= delta[0];
			beta_x -= delta[1];
			beta_y -= delta[2];
			beta_z -= delta[3];

		}
	}
}