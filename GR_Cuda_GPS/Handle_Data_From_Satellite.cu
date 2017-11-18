
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



__global__ void kernelMultiArray2D(short* ACos, short *ASin, short* B, short* CCos, short *CSin, int rows, int cols, short gain) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;
	int temp = ACos[index];
	ACos[index] = temp * B[index] * CCos[index] * gain;
	ASin[index] = temp * B[index] * CSin[index] * gain;
}

__global__ void kernerSumColumnArray2D(short *ACos, short *ASin, int rows, int cols, short *iq_buff) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;

	if (col < cols && row == 0) {
		int sumCos = 0;
		int sumSin = 0;
		for (int i = 0; i < rows; i++) {
			sumCos += ACos[i*cols + index];
			sumSin += ASin[i*cols + index];
		}
		iq_buff[col * 2] = short((sumCos + 64) >> 7);
		iq_buff[col * 2 + 1] = short((sumSin + 64) >> 7);
		//arrayCos[col] = sumCos;
		//arraySin[col] = sumSin;
	}
}


extern "C" void multiArray2D_Wrapper(short *h_A, short *h_B, short *h_CCos, short *h_CSin, int rows, int cols, short *arrayCos, short *arraySin, short *h_iq_buff, short gain) {

	size_t size_array1D = cols * sizeof(short);
	size_t size_array2D = rows * cols * sizeof(short);

	short *dev_ACos, *dev_ASin, *dev_B, *dev_CCos, *dev_CSin;
	//short *d_arrayCos, *d_arraySin;
	short *d_iq_buff;

	int blockSize = BLOCK_SIZES;
	int gridSize = (rows * cols + blockSize - 1) / blockSize;

	checkCuda(cudaMalloc((void**)&dev_ACos, size_array2D));
	checkCuda(cudaMalloc((void**)&dev_ASin, size_array2D));
	checkCuda(cudaMalloc((void**)&dev_B, size_array2D));
	checkCuda(cudaMalloc((void**)&dev_CCos, size_array2D));
	checkCuda(cudaMalloc((void**)&dev_CSin, size_array2D));
	//checkCuda(cudaMalloc((void**)&d_arrayCos, size_array1D));
	//checkCuda(cudaMalloc((void**)&d_arraySin, size_array1D));
	checkCuda(cudaMalloc((void**)&d_iq_buff, cols * 2 * sizeof(short)));


	checkCuda(cudaMemcpy(dev_ACos, h_A, size_array2D, cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpy(dev_ASin, h_ASin, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_B, h_B, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_CCos, h_CCos, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_CSin, h_CSin, size_array2D, cudaMemcpyHostToDevice));


	kernelMultiArray2D << <gridSize, blockSize >> >(dev_ACos, dev_ASin, dev_B, dev_CCos, dev_CSin, rows, cols, gain);
	kernerSumColumnArray2D << <gridSize, blockSize >> > (dev_ACos, dev_ASin, rows, cols, d_iq_buff);
	cudaThreadSynchronize();

	//checkCuda(cudaMemcpy(arrayCos, d_arrayCos, size_array1D, cudaMemcpyDeviceToHost));
	//checkCuda(cudaMemcpy(arraySin, d_arraySin, size_array1D, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_iq_buff, d_iq_buff, cols * 2 * sizeof(short), cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(dev_ACos));
	checkCuda(cudaFree(dev_ASin));
	checkCuda(cudaFree(dev_B));
	checkCuda(cudaFree(dev_CCos));
	checkCuda(cudaFree(dev_CSin));
	//checkCuda(cudaFree(d_arrayCos));
	//checkCuda(cudaFree(d_arraySin));

	printf("\nhet\n");
}