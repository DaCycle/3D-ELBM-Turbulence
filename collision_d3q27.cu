#include "d3q27.cuh"

__global__
void collision_d3q27(double* f, double* viscousity, double* f_new, double* f_eq, double Beta, int Cell_Count) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds

	for (int i = idx; i < Cell_Count; i += stride) {
		// Initialize Variables
		double z[27];
		double z_neg[27] = { 0.0 };
		double z_max = 0;
		double z_min = INFINITY;
		double f_neg[27] = { 0.0 };
		double a = 0;
		double b = 0;
		double c = 0;
		double alpha;
		int index;

		// Calculate z and z_neg values
		for (int d = 0; d < 27; d++) {
			index = i * 27 + d;
			z[d] = f_eq[index] / f_new[index] - 1;
			if (z[d] < 0) {
				z_neg[d] = z[d];
				f_neg[d] = f_new[index];
			}
			z_max = max(z_max, z[d]);
			z_min = min(z_min, z[d]);
		}

		// Calculate a, b, and c for quadratic equation
		for (int d = 0; d < 27; d++) {
			index = i * 27 + d;
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

		// Collision Step
		for (int d = 0; d < 27; d++) {
			index = i * 27 + d;
			f[index] = f_new[index] + alpha * Beta * (f_eq[index] - f_new[index]);
		}

		// Write Viscousity
		//viscousity[i] = (1.0 / alpha - 0.5) / 3.0;
		viscousity[i] = (1.0 / alpha / Beta - 0.5) / 3.0;
	}
}

//#include "d3q27.cuh"
//
//__global__
//void collision_d3q27(double* f, double* viscousity, double* f_new, double* f_eq, double Beta, int Cell_Count, int* Opp)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//
//    if (idx >= Cell_Count) return;
//
//    for (int i = idx; i < Cell_Count; i += stride) {
//
//        double z[27];
//        double z_neg[27] = { 0.0 };
//        double f_neg[27] = { 0.0 };
//
//        double z_max = 0.0;
//        double z_min = INFINITY;
//
//        double a = 0.0;
//        double b = 0.0;
//        double c = 0.0;
//
//        int index;
//
//        // ---- Entropic quantities (unchanged) ----
//        for (int d = 0; d < 27; d++) {
//            index = i * 27 + d;
//            z[d] = f_eq[index] / f_new[index] - 1.0;
//
//            if (z[d] < 0.0) {
//                z_neg[d] = z[d];
//                f_neg[d] = f_new[index];
//            }
//
//            z_max = max(z_max, z[d]);
//            z_min = min(z_min, z[d]);
//        }
//
//        for (int d = 0; d < 27; d++) {
//            index = i * 27 + d;
//            a += f_neg[d] * z_neg[d] * z_neg[d] * z_neg[d] * 0.5;
//            b += f_new[index] * z[d] * z[d] * 0.5;
//            c += f_new[index] * 2.0 * z[d] * z[d] / (2.0 + z[d]);
//        }
//
//        double denominator = b + sqrt(b * b - 4.0 * a * c);
//        double alpha = (denominator == 0.0) ? 2.0 : (2.0 * c / denominator);
//
//        // ---- TRT parameters ----
//        double omega_plus = alpha * Beta;
//
//        double omega_minus =
//            1.0 / (0.5 + 1.0 / (4.0 * (1.0 / omega_plus - 0.5)));
//
//        // ---- TRT collision ----
//        for (int d = 0; d < 27; d++) {
//
//            int db = Opp[d];
//
//            if (d > db) continue; // handle each pair once
//
//            int idx_d = i * 27 + d;
//            int idx_db = i * 27 + db;
//
//            double f_d = f_new[idx_d];
//            double f_db = f_new[idx_db];
//
//            double feq_d = f_eq[idx_d];
//            double feq_db = f_eq[idx_db];
//
//            // Symmetric / antisymmetric split
//            double f_plus = 0.5 * (f_d + f_db);
//            double f_minus = 0.5 * (f_d - f_db);
//
//            double feq_plus = 0.5 * (feq_d + feq_db);
//            double feq_minus = 0.5 * (feq_d - feq_db);
//
//            // Relax
//            f_plus -= omega_plus * (f_plus - feq_plus);
//            f_minus -= omega_minus * (f_minus - feq_minus);
//
//            // Recombine
//            f[idx_d] = f_plus + f_minus;
//            f[idx_db] = f_plus - f_minus;
//        }
//
//        // Rest particle (self-opposite)
//        int d0 = 0;
//        int idx0 = i * 27 + d0;
//        f[idx0] = f_new[idx0] + omega_plus * (f_eq[idx0] - f_new[idx0]);
//
//        // ---- Effective viscosity ----
//        viscousity[i] = (1.0 / omega_plus - 0.5) / 3.0;
//    }
//}
