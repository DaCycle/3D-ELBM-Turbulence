//#include "d3q27.cuh"
//
//__global__
//void streaming_d3q27(double* f_new, double* f, double U_lid,
//    int N_x, int Cell_Count, double* w, int* Ksi)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//
//    if (idx >= Cell_Count) return;
//
//    for (int index = idx; index < Cell_Count; index += stride) {
//
//        int i = index % N_x;
//        int j = (index / N_x) % N_x;
//        int k = index / (N_x * N_x);
//
//        double Rho_t = 0.0;
//        double A = 0.0;
//
//        // ==============================
//        // Streaming + density collection
//        // ==============================
//        for (int d = 0; d < 27; d++) {
//
//            int ex = Ksi[3 * d];
//            int ey = Ksi[3 * d + 1];
//            int ez = Ksi[3 * d + 2];
//
//            int in = i - ex;
//            int jn = j - ey;
//            int kn = k - ez;
//
//            bool inside =
//                (in >= 0 && in < N_x) &&
//                (jn >= 0 && jn < N_x) &&
//                (kn >= 0 && kn < N_x);
//
//            if (inside) {
//                f_new[index * 27 + d] =
//                    f[(kn * N_x * N_x + jn * N_x + in) * 27 + d];
//                Rho_t += f_new[index * 27 + d];
//            }
//            else {
//                // ------------------------------
//                // Direction-dependent wall speed
//                // ------------------------------
//                bool hit_x_wall = (in < 0 || in >= N_x);
//                bool hit_y_wall = (jn < 0 || jn >= N_x);
//                bool hit_z_wall = (kn < 0 || kn >= N_x);
//
//                double ux_bc = 0.0;
//
//                // Lid OR corner touching lid
//                if (hit_z_wall && k == N_x - 1) {
//                    ux_bc = U_lid;
//                }
//                // Side walls explicitly stay ux = 0
//
//                double cu = 3.0 * (ex * ux_bc);
//                double uu = 1.5 * ux_bc * ux_bc;
//
//                A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//            }
//        }
//
//        // ==============================
//        // Diffuse density
//        // ==============================
//        Rho_t = Rho_t / (1.0 - A);
//
//        // ==============================
//        // Assign missing PDFs
//        // ==============================
//        if (i == 0 || i == N_x - 1 || j == 0 || j == N_x - 1 || k == 0 || k == N_x - 1) {
//
//            for (int d = 0; d < 27; d++) {
//
//                int ex = Ksi[3 * d];
//                int ey = Ksi[3 * d + 1];
//                int ez = Ksi[3 * d + 2];
//
//                int in = i - ex;
//                int jn = j - ey;
//                int kn = k - ez;
//
//                bool inside =
//                    (in >= 0 && in < N_x) &&
//                    (jn >= 0 && jn < N_x) &&
//                    (kn >= 0 && kn < N_x);
//
//                if (!inside) {
//
//                    bool hit_x_wall = (in < 0 || in >= N_x);
//                    bool hit_y_wall = (jn < 0 || jn >= N_x);
//                    bool hit_z_wall = (kn < 0 || kn >= N_x);
//
//                    double ux_bc = 0.0;
//
//                    // Lid and lid-corners
//                    if (hit_z_wall && k == N_x - 1) {
//                        ux_bc = U_lid;
//                    }
//
//                    double cu = 3.0 * (ex * ux_bc);
//                    double uu = 1.5 * ux_bc * ux_bc;
//
//                    f_new[index * 27 + d] =
//                        Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//                }
//            }
//        }
//    }
//}

#include "d3q27.cuh"

__global__
void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds

	for (int index = idx; index < Cell_Count; index += stride) {
		int i = index % N_x;
		int j = (index / N_y) % N_x;
		int k = index / (N_x * N_y);

		double Rho_t = 0.0; // Temporary density variable
		double A = 0.0; // Constant for Maxwell Diffuse Boundary Conditions

		// Compute Wall Velocity
		double ux = 0.0;
		if (k == N_x-1) { ux = U_lid; }

		// Interior and Known Boundary Nodes
		for (int d = 0; d < 27; d++) {			
			int in = i - Ksi[3 * d];
			int jn = j - Ksi[3 * d + 1];
			int kn = k - Ksi[3 * d + 2];

			bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);

			if (inside) {
				f_new[index * 27 + d] = f[(kn*N_x*N_y + jn*N_y + in) * 27 + d];
				Rho_t += f_new[index * 27 + d];
			}
			else {
				double cu = 3.0 * (Ksi[3 * d] * ux);
				double uu = 1.5 * (ux * ux);
				A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
			}
		}

		// Compute Diffuse Density
		Rho_t = Rho_t / (1.0 - A);

		// Assign Missing PDFs using Maxwellian Diffuse BC
		if (i == 0 || i == N_x-1 || j == 0 || j == N_y-1 || k == 0 || k == N_x-1) {
			for (int d = 0; d < 27; d++) {
				int in = i - Ksi[3 * d];
				int jn = j - Ksi[3 * d + 1];
				int kn = k - Ksi[3 * d + 2];

				bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);

				if (!inside) {
					double cu = 3.0 * (Ksi[3 * d] * ux);
					double uu = 1.5 * (ux * ux);
					f_new[index * 27 + d] = Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
				}
			}
		}
	}
}