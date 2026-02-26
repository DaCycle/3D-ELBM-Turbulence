//#include "d3q27.cuh"
//
//__global__
//void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds
//
//	for (int index = idx; index < Cell_Count; index += stride) {
//		int i = index % N_x;
//		int j = (index / N_y) % N_x;
//		int k = index / (N_x * N_y);
//
//		double Rho_t = 0.0; // Temporary density variable
//		double A = 0.0; // Constant for Maxwell Diffuse Boundary Conditions
//
//		// Compute Wall Velocity
//		double ux = 0.0;
//		if (k == N_x-1) { ux = U_lid; }
//
//		// Interior and Known Boundary Nodes
//		for (int d = 0; d < 27; d++) {			
//			int in = i - Ksi[3 * d];
//			int jn = j - Ksi[3 * d + 1];
//			int kn = k - Ksi[3 * d + 2];
//
//			bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);
//
//			if (inside) {
//				f_new[index * 27 + d] = f[(kn*N_x*N_y + jn*N_y + in) * 27 + d];
//				Rho_t += f_new[index * 27 + d];
//			}
//			else {
//				double cu = 3.0 * (Ksi[3 * d] * ux);
//				double uu = 1.5 * (ux * ux);
//				A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//			}
//		}
//
//		// Compute Diffuse Density
//		Rho_t = Rho_t / (1.0 - A);
//
//		// Assign Missing PDFs using Maxwellian Diffuse BC
//		if (i == 0 || i == N_x-1 || j == 0 || j == N_y-1 || k == 0 || k == N_x-1) {
//			for (int d = 0; d < 27; d++) {
//				int in = i - Ksi[3 * d];
//				int jn = j - Ksi[3 * d + 1];
//				int kn = k - Ksi[3 * d + 2];
//
//				bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);
//
//				if (!inside) {
//					double cu = 3.0 * (Ksi[3 * d] * ux);
//					double uu = 1.5 * (ux * ux);
//					f_new[index * 27 + d] = Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//				}
//			}
//		}
//	}
//}

// Half Way Bounce-Back
//#include "d3q27.cuh"   // must contain __constant__ int cx[27], cy[27], cz[27], opp[27];  and __constant__ double w[27];
//
//__global__
//void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi, int*opp)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//
//    for (int i = idx; i < Cell_Count; i += stride) {
//        int x = i % N_x;
//        int y = (i / N_x) % N_y;
//        int z = i / (N_x * N_y);
//
//        bool is_lid = (z == N_x - 1);
//
//        // rho only needed on the lid
//        double rho = 0.0;
//        if (is_lid) {
//            for (int d = 0; d < 27; d++) {
//                rho += f[i * 27 + d];
//            }
//        }
//
//        for (int d = 0; d < 27; d++) {
//            int sx = x - Ksi[3 * d];
//            int sy = y - Ksi[3 * d + 1];
//            int sz = z - Ksi[3 * d + 2];
//
//            if (sx >= 0 && sx < N_x && sy >= 0 && sy < N_y && sz >= 0 && sz < N_x) {
//                // normal streaming from inside the domain
//                int sidx = sx + sy * N_x + sz * (N_x * N_y);
//                f_new[i * 27 + d] = f[sidx * 27 + d];
//            }
//            else {
//                // boundary crossed → half-way bounce-back
//                int op = opp[d];
//                double val = f[i * 27 + op];
//
//                // moving-lid correction only when the link crossed the lid (sz >= N_x)
//                if (is_lid && sz >= N_x) {
//                    val += 6.0 * w[d] * rho * (Ksi[3 * d] * U_lid);   // 6 = 2 / cs2
//                }
//                // all other walls (bottom, sides, front/back) are stationary → no extra term
//
//                f_new[i * 27 + d] = val;
//            }
//        }
//    }
//}

// Half Way Bounce-Back for all boundaries except top lid, Maxwell Diffuse for top lid
#include "d3q27.cuh"

__global__
void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi, int* Opp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (idx >= Cell_Count) return;

    for (int index = idx; index < Cell_Count; index += stride) {

        int i = index % N_x;
        int j = (index / N_x) % N_y;
        int k = index / (N_x * N_y);

        bool on_x_wall = (i == 0 || i == N_x - 1);
        bool on_y_wall = (j == 0 || j == N_y - 1);
        bool on_z_wall = (k == 0);
        bool on_lid = (k == N_x - 1);

        bool stationary_wall = (on_x_wall || on_y_wall || on_z_wall) && !on_lid;

        double Rho_t = 0.0;
        double A = 0.0;

        // Lid velocity
        double ux = (on_lid ? U_lid : 0.0);

        // --- Streaming ---
        for (int d = 0; d < 27; d++) {

            int in = i - Ksi[3 * d];
            int jn = j - Ksi[3 * d + 1];
            int kn = k - Ksi[3 * d + 2];

            bool inside =
                (in >= 0 && in < N_x) &&
                (jn >= 0 && jn < N_y) &&
                (kn >= 0 && kn < N_x);

            if (inside) {
                f_new[index * 27 + d] =
                    f[(kn * N_x * N_y + jn * N_x + in) * 27 + d];
                Rho_t += f_new[index * 27 + d];
            }
            else {

                // --- Stationary walls: half-way bounce-back ---
                if (stationary_wall) {
                    int db = Opp[d];
                    f_new[index * 27 + d] = f[index * 27 + db];
                }
                // --- Lid / edges touching lid: diffuse ---
                else {
                    double cu = 3.0 * (Ksi[3 * d] * ux);
                    double uu = 1.5 * ux * ux;
                    A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
                }
            }
        }

        // --- Diffuse density (only needed on lid / lid edges) ---
        if (!stationary_wall && (on_x_wall || on_y_wall || on_lid)) {
            Rho_t /= (1.0 - A);
        }

        // --- Assign missing PDFs for diffuse nodes only ---
        if (!stationary_wall && (on_x_wall || on_y_wall || on_lid)) {
            for (int d = 0; d < 27; d++) {

                int in = i - Ksi[3 * d];
                int jn = j - Ksi[3 * d + 1];
                int kn = k - Ksi[3 * d + 2];

                bool inside =
                    (in >= 0 && in < N_x) &&
                    (jn >= 0 && jn < N_y) &&
                    (kn >= 0 && kn < N_x);

                if (!inside) {
                    double cu = 3.0 * (Ksi[3 * d] * ux);
                    double uu = 1.5 * ux * ux;
                    f_new[index * 27 + d] =
                        Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
                }
            }
        }
    }
}


//#include "d3q27.cuh"
//
//__global__
//void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi, int* opp) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	if (idx >= Cell_Count) return; // Ensure we don't access out of bounds
//
//	for (int index = idx; index < Cell_Count; index += stride) {
//		int i = index % N_x;
//		int j = (index / N_y) % N_x;
//		int k = index / (N_x * N_y);
//
//		bool use_diffuse = (k == N_x - 1);
//		double ux = use_diffuse ? U_lid : 0.0;
//		bool is_other_boundary = (i == 0 || i == N_x - 1 || j == 0 || j == N_y - 1 || k == 0);
//
//		double Rho_t = 0.0;
//		double A = 0.0;
//
//		// First pass: stream known distributions
//		for (int d = 0; d < 27; d++) {
//			int in = i - Ksi[3 * d];
//			int jn = j - Ksi[3 * d + 1];
//			int kn = k - Ksi[3 * d + 2];
//
//			bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);
//
//			if (inside) {
//				f_new[index * 27 + d] = f[(kn * N_x * N_y + jn * N_y + in) * 27 + d];
//				if (use_diffuse) {
//					Rho_t += f_new[index * 27 + d];
//				}
//			}
//			else if (use_diffuse) {
//				double cu = 3.0 * (Ksi[3 * d] * ux);
//				double uu = 1.5 * (ux * ux);
//				A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//			}
//		}
//
//		// Apply diffuse BC if on top
//		if (use_diffuse) {
//			if (1.0 - A > 1e-10) {
//				Rho_t /= (1.0 - A);
//			} // else Rho_t remains 0 or handle as needed
//
//			for (int d = 0; d < 27; d++) {
//				int in = i - Ksi[3 * d];
//				int jn = j - Ksi[3 * d + 1];
//				int kn = k - Ksi[3 * d + 2];
//
//				bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);
//
//				if (!inside) {
//					double cu = 3.0 * (Ksi[3 * d] * ux);
//					double uu = 1.5 * (ux * ux);
//					f_new[index * 27 + d] = Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//				}
//			}
//		}
//		else if (is_other_boundary) {
//			// Apply bounce-back for other boundaries
//			for (int d = 0; d < 27; d++) {
//				int in = i - Ksi[3 * d];
//				int jn = j - Ksi[3 * d + 1];
//				int kn = k - Ksi[3 * d + 2];
//
//				bool inside = (in >= 0) && (in < N_x) && (jn >= 0) && (jn < N_y) && (kn >= 0) && (kn < N_x);
//
//				if (!inside) {
//					int opp_d = opp[d];
//					f_new[index * 27 + d] = f[index * 27 + opp_d];
//				}
//			}
//		}
//	}
//}

//#include "d3q27.cuh"
//__global__
//void streaming_d3q27(double* f_new, double* f, double U_lid, int N_x, int N_y, int Cell_Count, double* w, int* Ksi, int* opp) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    if (idx >= Cell_Count) return;
//
//    for (int index = idx; index < Cell_Count; index += stride) {
//        int i = index % N_x;
//        int j = (index / N_x) % N_y;  // Corrected: assuming N_y may differ from N_x, but typically square
//        int k = index / (N_x * N_y);
//
//        bool is_top_lid = (k == N_x - 1);
//        bool is_other_boundary = (i == 0 || i == N_x - 1 || j == 0 || j == N_y - 1 || k == 0);
//
//        double ux = is_top_lid ? U_lid : 0.0;
//        double uy = 0.0;
//        double uz = 0.0;
//
//        double rho = 0.0;
//
//        // First pass: stream known distributions and accumulate rho from known
//        for (int d = 0; d < 27; d++) {
//            int in = i - Ksi[3 * d];
//            int jn = j - Ksi[3 * d + 1];
//            int kn = k - Ksi[3 * d + 2];
//            bool inside = (in >= 0 && in < N_x) && (jn >= 0 && jn < N_y) && (kn >= 0 && kn < N_x);
//
//            if (inside) {
//                f_new[index * 27 + d] = f[(kn * N_x * N_y + jn * N_x + in) * 27 + d];  // Corrected: N_y to N_x in jn term?
//                rho += f_new[index * 27 + d];
//            }  // unknowns remain unset (assume initialized to 0)
//        }
//
//        // Apply diffuse BC if on top lid
//        if (is_top_lid) {
//            double A = 0.0;
//            for (int d = 0; d < 27; d++) {
//                int in = i - Ksi[3 * d];
//                int jn = j - Ksi[3 * d + 1];
//                int kn = k - Ksi[3 * d + 2];
//                bool inside = (in >= 0 && in < N_x) && (jn >= 0 && jn < N_y) && (kn >= 0 && kn < N_x);
//                if (!inside) {
//                    double cu = 3.0 * (Ksi[3 * d] * ux + Ksi[3 * d + 1] * uy + Ksi[3 * d + 2] * uz);
//                    double uu = 1.5 * (ux * ux + uy * uy + uz * uz);
//                    A += w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//                }
//            }
//            double Rho_t = (1.0 - A > 1e-10) ? rho / (1.0 - A) : 0.0;  // handle division
//            for (int d = 0; d < 27; d++) {
//                int in = i - Ksi[3 * d];
//                int jn = j - Ksi[3 * d + 1];
//                int kn = k - Ksi[3 * d + 2];
//                bool inside = (in >= 0 && in < N_x) && (jn >= 0 && jn < N_y) && (kn >= 0 && kn < N_x);
//                if (!inside) {
//                    double cu = 3.0 * (Ksi[3 * d] * ux + Ksi[3 * d + 1] * uy + Ksi[3 * d + 2] * uz);
//                    double uu = 1.5 * (ux * ux + uy * uy + uz * uz);
//                    f_new[index * 27 + d] = Rho_t * w[d] * (1.0 + cu + 0.5 * cu * cu - uu);
//                }
//            }
//        }
//        else if (is_other_boundary) {
//            // Apply Zou-He for stationary walls (u_b = 0, reduces to bounce-back with post-stream values)
//            ux = 0.0; uy = 0.0; uz = 0.0;  // explicit for clarity
//            for (int d = 0; d < 27; d++) {
//                int in = i - Ksi[3 * d];
//                int jn = j - Ksi[3 * d + 1];
//                int kn = k - Ksi[3 * d + 2];
//                bool inside = (in >= 0 && in < N_x) && (jn >= 0 && jn < N_y) && (kn >= 0 && kn < N_x);
//                if (!inside) {
//                    int opp_d = opp[d];
//                    double cu = 3.0 * (Ksi[3 * d] * ux + Ksi[3 * d + 1] * uy + Ksi[3 * d + 2] * uz);  // 0
//                    f_new[index * 27 + d] = f_new[index * 27 + opp_d] + 6.0 * w[d] * rho * (cu / 3.0);
//                }
//            }
//        }
//    }
//}