#include "d3q27.cuh"
#include "file_write.h"

__constant__ double w[9];
__constant__ double Ksi[9][2];

int main()
{
	// Tuner
	bool start_from_new = true;

	const int N_x = 128;
	double Beta = 0.99559;
	double Re = 5000.0;	// Reynolds number
	int Timer = 10000;
	int FW_freq = 1000;

	// Definition of Parameters
	const int N_y = N_x;

	double h_w[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
	cudaMemcpyToSymbol(w, h_w, sizeof(h_w));

	double h_Ksi[9][2] = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { -1.0, 0.0 }, { 0.0, -1.0 }, { 1.0, 1.0 }, { -1.0, 1.0 }, { -1.0, -1.0 }, { 1.0, -1.0 } };
	cudaMemcpyToSymbol(Ksi, h_Ksi, sizeof(h_Ksi));

	const double U_lid = Re * (0.5 / Beta - 0.5) / (N_x * 3.0);; // Lid velocity

	// Initialization of Parameters
	double Rho_ref = 2.0; // Reference density

	double* h_Rho = new double[N_x * N_y];
	double* d_Rho;
	cudaMalloc((void**)&d_Rho, N_x * N_y * sizeof(double));

	double* h_U = new double[N_x * N_y * 2];
	double* d_U;
	cudaMalloc((void**)&d_U, N_x * N_y * 2 * sizeof(double));

	double* h_curl = new double[N_x * N_y];
	double* d_curl;
	cudaMalloc((void**)&d_curl, N_x * N_y * sizeof(double));

	double* h_viscousity = new double[N_x * N_y];
	double* d_viscousity;
	cudaMalloc((void**)&d_viscousity, N_x * N_y * sizeof(double));

	double* h_f = new double[N_x * N_y * 9];
	double* d_f;
	cudaMalloc((void**)&d_f, N_x * N_y * 9 * sizeof(double));

	double* h_f_eq = new double[N_x * N_y * 9];
	double* d_f_eq;
	cudaMalloc((void**)&d_f_eq, N_x * N_y * 9 * sizeof(double));

	double* h_f_new = new double[N_x * N_y * 9];
	double* d_f_new;
	cudaMalloc((void**)&d_f_new, N_x * N_y * 9 * sizeof(double));

	double* h_Rho_BC = new double[N_x * N_y];
	double* d_Rho_BC;
	cudaMalloc((void**)&d_Rho_BC, N_x * N_y * sizeof(double));

	double* h_U_BC = new double[N_x * N_y * 2];
	double* d_U_BC;
	cudaMalloc((void**)&d_U_BC, N_x * N_y * 2 * sizeof(double));

	double* h_f_eq_BC = new double[N_x * N_y * 9];
	double* d_f_eq_BC;
	cudaMalloc((void**)&d_f_eq_BC, N_x * N_y * 9 * sizeof(double));

	int blockSize = 256;
	int smallBlocks = (N_x * N_y + blockSize - 1) / blockSize;
	//int bigBlocks = (N_x * N_y * 9 + blockSize - 1) / blockSize;

	// Initialize Diffuse BC
	const double A_TL = (2.0 - 1.5 * U_lid * U_lid) / 9.0 + (3.0 + 6.0 * U_lid * U_lid) / 36;
	const double A_TR = (2.0 - 1.5 * U_lid * U_lid) / 9.0 + (3.0 + 6.0 * U_lid * U_lid) / 36;
	const double A_TM = (1 - 1.5 * U_lid * U_lid) / 9.0 + (1 + 3.0 * U_lid * U_lid) / 18.0;

	// File Writing
	const string outputFile = "snapshots";

	if (start_from_new)
	{
		createOutputFolder(outputFile);

		for (int i = 0; i < N_x * N_y; i++)
		{
			h_Rho[i] = Rho_ref;
			h_Rho_BC[i] = 1.0;
		}
		cudaMemcpy(d_Rho, h_Rho, N_x * N_y * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Rho_BC, h_Rho_BC, N_x * N_y * sizeof(double), cudaMemcpyHostToDevice);
		for (int i = 0; i < N_x * N_y * 2; i++)
		{
			if (i < N_x) {
				h_U_BC[i] = U_lid;
				h_U[i] = U_lid;
			}
			else {
				h_U_BC[i] = 0.0;
				h_U[i] = 0.0;
			}
		}
		cudaMemcpy(d_U, h_U, N_x * N_y * 2 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_U_BC, h_U_BC, N_x * N_y * 2 * sizeof(double), cudaMemcpyHostToDevice);

		eqm_d3q27 << <smallBlocks, blockSize >> > (d_f_eq_BC, d_Rho_BC, d_U_BC, N_x, N_y);

		eqm_d3q27 << <smallBlocks, blockSize >> > (d_f, d_Rho, d_U, N_x, N_y);

		cudaDeviceSynchronize();
		cudaMemcpy(d_f_eq, d_f, N_x * N_y * 9 * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_f_new, d_f, N_x * N_y * 9 * sizeof(double), cudaMemcpyDeviceToDevice);

	}
	else {
		// Do later
	}

	// Solving
	auto start = std::chrono::high_resolution_clock::now();
	for (int t = 0; t < Timer; t++)
	{
		// Streaming/Boundary Conditions
		streaming_d3q27 << <smallBlocks, blockSize >> > (d_f_new, d_f, d_f_eq_BC, d_Rho, U_lid, A_TL, A_TR, A_TM, N_x, N_y);

		// Moment Calculation
		moment_rho_u_d3q27 << <smallBlocks, blockSize >> > (d_Rho, d_U, d_f_new, N_x, N_y);

		// Equilibrium Calculation
		eqm_d3q27 << <smallBlocks, blockSize >> > (d_f_eq, d_Rho, d_U, N_x, N_y);

		// Collision
		collision_d3q27 << <smallBlocks, blockSize >> > (d_f, d_viscousity, d_f_new, d_f_eq, Beta, N_x, N_y);

		// File Writing
		if ((t + 1) % FW_freq == 0) {
			curl_2D << <smallBlocks, blockSize >> > (d_curl, d_U, N_x, N_y);
			cudaDeviceSynchronize();

			cudaMemcpy(h_Rho, d_Rho, N_x * N_y * sizeof(double), cudaMemcpyDeviceToHost);
			writeToFile(t + 1, h_Rho, "Density", outputFile, N_x * N_y * sizeof(double));

			cudaMemcpy(h_U, d_U, N_x * N_y * 2 * sizeof(double), cudaMemcpyDeviceToHost);
			writeToFile(t + 1, h_U, "Velocity", outputFile, N_x * N_y * 2 * sizeof(double));

			cudaMemcpy(h_curl, d_curl, N_x * N_y * sizeof(double), cudaMemcpyDeviceToHost);
			writeToFile(t + 1, h_curl, "Curl", outputFile, N_x * N_y * sizeof(double));

			cudaMemcpy(h_viscousity, d_viscousity, N_x * N_y * sizeof(double), cudaMemcpyDeviceToHost);
			writeToFile(t + 1, h_viscousity, "Viscousity", outputFile, N_x * N_y * sizeof(double));
		}
	}

	// Post-processing
	cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> Runtime = stop - start;
	cout << "Runtime: " << Runtime.count() << " seconds" << "\n";
	cout << "Runtime per iteration per node: " << Runtime.count() / Timer / (N_x * N_y) << " seconds" << "\n";

	// Memory cleanup
	cudaFree(d_Rho);
	cudaFree(d_U);
	cudaFree(d_f);
	cudaFree(d_f_eq);
	cudaFree(d_f_new);
	cudaFree(d_curl);
	cudaFree(d_viscousity);
	cudaFree(d_Rho_BC);
	cudaFree(d_U_BC);
	cudaFree(d_f_eq_BC);
	delete[] h_Rho;
	delete[] h_U;
	delete[] h_f;
	delete[] h_f_eq;
	delete[] h_f_new;
	delete[] h_curl;
	delete[] h_viscousity;
	delete[] h_Rho_BC;
	delete[] h_U_BC;
	delete[] h_f_eq_BC;

	return 0;
}