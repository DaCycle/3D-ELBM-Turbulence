#include "d3q27.cuh"

__global__
void eqm_d3q27(double* f_eq, double* Rho, double* U, int N_x, int N_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= N_x * N_y) return; // Ensure we don't access out of bounds

	//for (int i = idx; i < N_x * N_y * 9; i += stride) {
	//	int I = i / 9; // Cell index
	//	int d = i % 9; // Direction index
	//
	//	double U_x = U[I];
	//	double U_y = U[I + N_x * N_y];
	//	double Ksi_U = __fma_rn(Ksi[d][0], U_x, Ksi[d][1] * U_y);
	//	double UU = __fma_rn(U_x, U_x, U_y * U_y);
	//
	//	f_eq[i] = w[d] * Rho[I] * __fma_rn(-1.5, UU, __fma_rn(4.5, Ksi_U * Ksi_U, __fma_rn(3.0, Ksi_U, 1.0)));
	//}

	if (idx >= N_x * N_y) return; // Ensure we don't access out of bounds

	for (int i = idx; i < N_x * N_y; i += stride) {

		// Entropic Lattice Boltzmann method equilibrium distribution function
		double U_x = U[i];
		double U_y = U[i + N_x * N_y];
		double UU = __fma_rn(U_x, U_x, U_y * U_y);

		// Initial guess for lagrangian multipliers and define Residuals
		double alpha = std::log(Rho[i]) - 1.5 * UU;
		double beta_x = 3.0 * U_x;
		double beta_y = 3.0 * U_y;

		// Initialize variables for Newton's method
		double R_0 = 1.0;
		double R_x = R_0;
		double R_y = R_0;
		double J_11;
		double J_12;
		double J_13;
		double J_22;
		double J_23;
		double J_33;
		double L_11;
		double L_12;
		double L_13;
		double L_22;
		double L_23;
		double L_33;
		double y1;
		double y2;
		double y3;
		double delta_alpha;
		double delta_beta_x;
		double delta_beta_y;

		// Newton's method loop
		for (int j = 0; j < 100; j++) {
			// Calculate this iteration's f_eq
			for (int d = 0; d < 9; d++) {
				f_eq[i * 9 + d] = w[d] * exp(alpha + beta_x * Ksi[d][0] + beta_y * Ksi[d][1]);
			}

			// Calculate Residuals
			J_11 = f_eq[i * 9 + 0] + f_eq[i * 9 + 1] + f_eq[i * 9 + 2] + f_eq[i * 9 + 3] + f_eq[i * 9 + 4] + f_eq[i * 9 + 5] + f_eq[i * 9 + 6] + f_eq[i * 9 + 7] + f_eq[i * 9 + 8];
			J_12 = f_eq[i * 9 + 1] - f_eq[i * 9 + 3] + f_eq[i * 9 + 5] - f_eq[i * 9 + 6] - f_eq[i * 9 + 7] + f_eq[i * 9 + 8];
			J_13 = f_eq[i * 9 + 2] - f_eq[i * 9 + 4] + f_eq[i * 9 + 5] + f_eq[i * 9 + 6] - f_eq[i * 9 + 7] - f_eq[i * 9 + 8];
			R_0 = J_11 - Rho[i];
			R_x = J_12 - U_x * Rho[i];
			R_y = J_13 - U_y * Rho[i];

			if (fabs(R_0) < 1e-4 && fabs(R_x) < 1e-4 && fabs(R_y) < 1e-4) {
				break;
			}

			// Calculate rest of Jacobian
			J_22 = f_eq[i * 9 + 1] + f_eq[i * 9 + 3] + f_eq[i * 9 + 5] + f_eq[i * 9 + 6] + f_eq[i * 9 + 7] + f_eq[i * 9 + 8];
			J_23 = f_eq[i * 9 + 5] - f_eq[i * 9 + 6] + f_eq[i * 9 + 7] - f_eq[i * 9 + 8];
			J_33 = f_eq[i * 9 + 2] + f_eq[i * 9 + 4] + f_eq[i * 9 + 5] + f_eq[i * 9 + 6] + f_eq[i * 9 + 7] + f_eq[i * 9 + 8];

			// Cholesky decomposition to avoid inverse of Jacobian
			L_11 = sqrt(J_11);
			L_12 = J_12 / L_11;
			L_13 = J_13 / L_11;
			L_22 = sqrt(J_22 - L_12 * L_12);
			L_23 = (J_23 - L_12 * L_13) / L_22;
			L_33 = sqrt(J_33 - L_13 * L_13 - L_23 * L_23);

			y1 = R_0 / L_11;
			y2 = (R_x - L_12 * y1) / L_22;
			y3 = (R_y - L_13 * y1 - L_23 * y2) / L_33;

			delta_beta_y = y3 / L_33;
			delta_beta_x = (y2 - L_23 * delta_beta_y) / L_22;
			delta_alpha = (y1 - L_12 * delta_beta_x - L_13 * delta_beta_y) / L_11;

			// Update lagrangian multipliers
			alpha -= delta_alpha;
			beta_x -= delta_beta_x;
			beta_y -= delta_beta_y;
		}
	}
}