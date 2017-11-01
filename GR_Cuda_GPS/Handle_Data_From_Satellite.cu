
#include "Handle_Data_From_Satellite.h"


//#define SIZE 1024
#define BLOCK_SIZES 32
#define GRID_SIZES 256

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		getchar();
		//assert(result == cudaSuccess);
	}
#endif
	return result;
}



__global__ void kernelMultiArray2D(short* A, short* B, short* C, short* D, size_t pitch, int rows, int cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = pitch / sizeof(short);
	int index_array = row * index + col;

	if ((col < cols) && (row < rows)) {
		A[index_array] = A[index_array] * B[index_array] * C[index_array] * D[index_array];
	}

}

__global__ void kernerSumColumnArray2D(short *A, short *array, size_t pitch, int rows, int cols) {

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < cols && row == 0) {
		short sum = 0;
		for (int i = 0; i < rows; i++) {
			short *temp_array = (short*)((char*)A + (row + i) * pitch);
			sum += temp_array[col];
		}
		array[col] = sum;
	}
}



extern "C" void multiArray2D_Wrapper(short *h_A, short *h_B, short *h_C, short *h_D, int rows, int cols, short *array) {

	size_t size_array1D = cols * sizeof(short);
	size_t size_array2D = rows * cols * sizeof(short);

	short *d_A, *d_B, *d_C, *d_D;
	short *d_array;
	size_t pitch;

	//array = (short *)malloc(size_array1D * sizeof(short));

	checkCuda(cudaMallocPitch((void**)&d_A, &pitch, cols * sizeof(short), rows));
	checkCuda(cudaMallocPitch((void**)&d_B, &pitch, cols * sizeof(short), rows));
	checkCuda(cudaMallocPitch((void**)&d_C, &pitch, cols * sizeof(short), rows));
	checkCuda(cudaMallocPitch((void**)&d_D, &pitch, cols * sizeof(short), rows));
	checkCuda(cudaMallocPitch((void**)&d_array, &pitch, cols * sizeof(short), 1));

	checkCuda(cudaMemcpy2D(d_A, pitch, h_A, cols * sizeof(short), cols * sizeof(short), rows, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy2D(d_B, pitch, h_B, cols * sizeof(short), cols * sizeof(short), rows, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy2D(d_C, pitch, h_C, cols * sizeof(short), cols * sizeof(short), rows, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy2D(d_D, pitch, h_D, cols * sizeof(short), cols * sizeof(short), rows, cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(1024, 1, 1);
	dim3 threadsPerBlock(1024, 1, 1);

	kernelMultiArray2D << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, d_D, pitch, rows, cols);
	kernerSumColumnArray2D << <blocksPerGrid, threadsPerBlock >> > (d_A, d_array, pitch, rows, cols);

	cudaThreadSynchronize();

	checkCuda(cudaMemcpy2D(array, cols * sizeof(short), d_array, pitch, cols * sizeof(short), 1, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));
	checkCuda(cudaFree(d_D));
	checkCuda(cudaFree(d_array));
	printf("\nhet\n");
}