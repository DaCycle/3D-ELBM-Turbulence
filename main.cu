#include "d3q27.cuh"
#include "file_write.h"

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        printf("CUDA error %s:%d: %s\n",                      \
               __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(1);                                              \
    }                                                         \
} while (0)


int main()
{
	// Tuner
	bool start_from_new = true;

	static constexpr int N_x = 81;
	const double U_lid = 0.05 / sqrt(3);// Re * (0.5 / Beta - 0.5) / (double(N_x) * 3.0); // Lid velocity
	double Re = 5000.0;	// Reynolds number
	double Beta = 1 / (6.0 * U_lid * double(N_x) / Re + 1);
	int Timer = 100000;
	int FW_freq = 1000;

	// Definition of Parameters
	static constexpr int N_y = N_x;
	static constexpr int N_z = N_y;
	static constexpr int Cell_Count = N_x * N_y * N_z;

	static constexpr int Q = 27; // D3Q27

	double h_w[Q] = { 8.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 
					  1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 
					  1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0 };
	double* w = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&w, Q * sizeof(double)));
	CUDA_CHECK(cudaMemcpy(w, h_w, Q * sizeof(double), cudaMemcpyHostToDevice));

	int h_Ksi[Q][3] = {
		// 0: rest
		{0,   0,  0},
		
		// 1-6: primary directions
		{ 1,  0,  0}, 
		{-1,  0,  0},
		{ 0,  1,  0}, 
		{ 0, -1,  0},
		{ 0,  0,  1}, 
		{ 0,  0, -1},
		
		// 7-18: secondary directions
		{ 1,  1,  0}, 
		{-1, -1,  0},
		{ 1, -1,  0}, 
		{-1,  1,  0},
		{ 1,  0,  1}, 
		{-1,  0, -1},
		{ 1,  0, -1}, 
		{-1,  0,  1},
		{ 0,  1,  1}, 
		{ 0, -1, -1},
		{ 0,  1, -1}, 
		{ 0, -1,  1},
		
		// 19-26: tertiary directions
		{ 1,  1,  1}, 
		{-1, -1, -1},
		{ 1,  1, -1}, 
		{-1, -1,  1},
		{ 1, -1,  1}, 
		{-1,  1, -1},
		{ 1, -1, -1}, 
		{-1,  1,  1}
	};
	int* Ksi = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&Ksi, Q * 3 * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(Ksi, h_Ksi, Q * 3 * sizeof(int), cudaMemcpyHostToDevice));

	// Initialization of Parameters
	double Rho_ref = 1.0; // Reference density

	double* h_Rho = new double[Cell_Count];
	double* d_Rho;
	CUDA_CHECK(cudaMalloc((void**)&d_Rho, Cell_Count * sizeof(double)));

	double* h_Ux = new double[Cell_Count];
	double* d_Ux;
	CUDA_CHECK(cudaMalloc((void**)&d_Ux, Cell_Count * sizeof(double)));

	double* h_Uy = new double[Cell_Count];
	double* d_Uy;
	CUDA_CHECK(cudaMalloc((void**)&d_Uy, Cell_Count * sizeof(double)));

	double* h_Uz = new double[Cell_Count];
	double* d_Uz;
	CUDA_CHECK(cudaMalloc((void**)&d_Uz, Cell_Count * sizeof(double)));

	double* h_viscousity = new double[Cell_Count];
	double* d_viscousity;
	CUDA_CHECK(cudaMalloc((void**)&d_viscousity, Cell_Count * sizeof(double)));

	double* h_f = new double[Cell_Count * Q];
	double* d_f;
	CUDA_CHECK(cudaMalloc((void**)&d_f, Cell_Count * Q * sizeof(double)));

	double* h_f_eq = new double[Cell_Count * Q];
	double* d_f_eq;
	CUDA_CHECK(cudaMalloc((void**)&d_f_eq, Cell_Count * Q * sizeof(double)));

	double* h_f_new = new double[Cell_Count * Q];
	double* d_f_new;
	CUDA_CHECK(cudaMalloc((void**)&d_f_new, Cell_Count * Q * sizeof(double)));

	int blockSize = 256;
	int smallBlocks = (Cell_Count + blockSize - 1) / blockSize;
	if (smallBlocks > 2147483647) {
		smallBlocks = 2147483647;
	}
	//int bigBlocks = (Cell_Count * 27 + blockSize - 1) / blockSize;

	// File Writing
	const string outputFile = "snapshots";

	if (start_from_new)
	{
		createOutputFolder(outputFile);

		for (int i = 0; i < Cell_Count; i++)
		{
			h_Rho[i] = Rho_ref;
			if (i / (N_x * N_y) == N_z - 1) {
				h_Ux[i] = U_lid;
			}
			else {
				h_Ux[i] = 0.0;
			}
			h_Uy[i] = 0.0;
			h_Uz[i] = 0.0;
		}
		CUDA_CHECK(cudaMemcpy(d_Rho, h_Rho, Cell_Count * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Ux, h_Ux, Cell_Count * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Uy, h_Uy, Cell_Count * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_Uz, h_Uz, Cell_Count * sizeof(double), cudaMemcpyHostToDevice));

		eqm_d3q27 << <smallBlocks, blockSize >> > (d_f, d_Rho, d_Ux, d_Uy, d_Uz, Cell_Count, w, Ksi);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(d_f_eq, d_f, Cell_Count * Q * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(d_f_new, d_f, Cell_Count * Q * sizeof(double), cudaMemcpyDeviceToDevice));

		CUDA_CHECK(cudaMemcpy(h_Rho, d_Rho, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
		writeToFile(0, h_Rho, "Density", outputFile, Cell_Count * sizeof(double));

		CUDA_CHECK(cudaMemcpy(h_Ux, d_Ux, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
		writeToFile(0, h_Ux, "X_Velocity", outputFile, Cell_Count * sizeof(double));

		CUDA_CHECK(cudaMemcpy(h_Uy, d_Uy, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
		writeToFile(0, h_Uy, "Y_Velocity", outputFile, Cell_Count * sizeof(double));

		CUDA_CHECK(cudaMemcpy(h_Uz, d_Uz, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
		writeToFile(0, h_Uz, "Z_Velocity", outputFile, Cell_Count * sizeof(double));

		CUDA_CHECK(cudaMemcpy(h_f, d_f, Cell_Count * Q * sizeof(double), cudaMemcpyDeviceToHost));
		writeToFile(0, h_f, "pdf", outputFile, Cell_Count * Q * sizeof(double));

	}
	else {
		// Do later
	}

	// Solving
	auto start = std::chrono::high_resolution_clock::now();
	for (int t = 0; t < Timer; t++)
	{
		// Streaming/Boundary Conditions
		streaming_d3q27 << <smallBlocks, blockSize >> > (d_f_new, d_f, U_lid, N_x, Cell_Count, w, Ksi);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Moment Calculation
		moment_rho_u_d3q27 << <smallBlocks, blockSize >> > (d_Rho, d_Ux, d_Uy, d_Uz, d_f_new, Cell_Count);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Equilibrium Calculation
		eqm_d3q27 << <smallBlocks, blockSize >> > (d_f_eq, d_Rho, d_Ux, d_Uy, d_Uz, Cell_Count, w, Ksi);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Collision
		collision_d3q27 << <smallBlocks, blockSize >> > (d_f, d_viscousity, d_f_new, d_f_eq, Beta, Cell_Count);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// File Writing
		if ((t + 1) % FW_freq == 0) {
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaMemcpy(h_Rho, d_Rho, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
			writeToFile(t + 1, h_Rho, "Density", outputFile, Cell_Count * sizeof(double));

			CUDA_CHECK(cudaMemcpy(h_Ux, d_Ux, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
			writeToFile(t + 1, h_Ux, "X_Velocity", outputFile, Cell_Count * sizeof(double));

			CUDA_CHECK(cudaMemcpy(h_Uy, d_Uy, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
			writeToFile(t + 1, h_Uy, "Y_Velocity", outputFile, Cell_Count * sizeof(double));

			CUDA_CHECK(cudaMemcpy(h_Uz, d_Uz, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost));
			writeToFile(t + 1, h_Uz, "Z_Velocity", outputFile, Cell_Count * sizeof(double));

			//cudaMemcpy(h_viscousity, d_viscousity, Cell_Count * sizeof(double), cudaMemcpyDeviceToHost);
			//writeToFile(t + 1, h_viscousity, "Viscousity", outputFile, Cell_Count * sizeof(double));
		}
	}

	// Post-processing
	CUDA_CHECK(cudaDeviceSynchronize());
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> Runtime = stop - start;
	cout << "Runtime: " << Runtime.count() << " seconds" << "\n";
	cout << "Runtime per iteration per node: " << Runtime.count() / Timer / (Cell_Count) << " seconds" << "\n";

	// Memory cleanup
	CUDA_CHECK(cudaFree(d_Rho));
	CUDA_CHECK(cudaFree(d_Ux));
	CUDA_CHECK(cudaFree(d_Uy));
	CUDA_CHECK(cudaFree(d_Uz));
	CUDA_CHECK(cudaFree(d_f));
	CUDA_CHECK(cudaFree(d_f_eq));
	CUDA_CHECK(cudaFree(d_f_new));
	CUDA_CHECK(cudaFree(d_viscousity));
	CUDA_CHECK(cudaFree(w));
	CUDA_CHECK(cudaFree(Ksi));
	delete[] h_Rho;
	delete[] h_Ux;
	delete[] h_Uy;
	delete[] h_Uz;
	delete[] h_f;
	delete[] h_f_eq;
	delete[] h_f_new;
	delete[] h_viscousity;

	return 0;
}