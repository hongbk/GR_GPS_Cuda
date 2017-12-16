
#include "Handle_Data_From_Satellite.h"

//__constant__  int sinTable512[] = {
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
//};
//
//__constant__ int cosTable512[] = {
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
//};



//#define SIZE 1024
#define BLOCK_SIZES 1024
#define GRID_SIZES 256

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s error code %d\n", cudaGetErrorString(result), result);
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

	}
}


extern "C" void multiArray2D_Wrapper(short *h_A, short *h_B, short *dev_CCos, short *dev_CSin, short* d_iq_buff, int rows, int cols, short *h_iq_buff, short gain) {

	size_t size_array1D = cols * sizeof(short);
	size_t size_array2D = rows * cols * sizeof(short);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int blockSize = BLOCK_SIZES;
	int gridSize = (rows * cols + blockSize - 1) / blockSize;


	checkCuda(cudaMemcpy(dev_CCos, h_A, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_CSin, h_B, size_array2D, cudaMemcpyHostToDevice));
	
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);

	//kernelMultiArray2D << <gridSize, blockSize >> >(dev_ACos, dev_ASin, dev_B, dev_CCos, dev_CSin, rows, cols, gain);
	kernerSumColumnArray2D << <gridSize, blockSize >> > (dev_CCos, dev_CSin, rows, cols, d_iq_buff);
	cudaThreadSynchronize();

	checkCuda(cudaMemcpy(h_iq_buff, d_iq_buff, cols * 2 * sizeof(short), cudaMemcpyDeviceToHost));

	printf("\nhet\n");
}

//__global__ void kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if (idx < iq_buff_size) {
//		int ip, qp, i_acc, q_acc;
//		int iTable;
//		i_acc = 0;
//		q_acc = 0;
//		for (int i = 0; i < count; i++) {
//			if (chan[i].prn > 0) {
//
//				iTable = (chan[i].carr_phase >> 16) & 511;
//
//				ip = chan[i].dataBit * chan[i].codeCA * cosTable512[iTable] * gain[i];
//				qp = chan[i].dataBit * chan[i].codeCA * sinTable512[iTable] * gain[i];
//
//				i_acc += ip;
//				q_acc += qp;
//
//				chan[i].code_phase += chan[i].f_code * delt;
//
//				if (chan[i].code_phase >= CA_SEQ_LEN) {
//
//					chan[i].code_phase -= CA_SEQ_LEN;
//					chan[i].icode++;
//
//					if (chan[i].icode >= 20) { // 20 C/A codes = 1 navigation data bit
//						chan[i].icode = 0;
//						chan[i].ibit++;
//
//						if (chan[i].ibit >= 30) { // 30 navigation data bits = 1 word
//							chan[i].ibit = 0;
//							chan[i].iword++;
//							/*
//							if (chan[i].iword>=N_DWRD)
//							printf("\nWARNING: Subframe word buffer overflow.\n");
//							*/
//						}
//
//						// Set new navigation data bit
//						chan[i].dataBit = (int)((chan[i].dwrd[chan[i].iword] >> (29 - chan[i].ibit)) & 0x1UL) * 2 - 1;
//					}
//				}
//
//				// Set currnt code chip
//				chan[i].codeCA = chan[i].ca[(int)chan[i].code_phase] <<1- 1;
//
//				// Update carrier phase
//				chan[i].carr_phase += chan[i].carr_phasestep;
//			}
//		}
//
//
//		// Scaled by 2^7
//		i_acc = (i_acc + 64) >> 7;
//		q_acc = (q_acc + 64) >> 7;
//
//		// Store I/Q samples into buffer
//		iq_buff[idx * 2] = (short)i_acc;
//		iq_buff[idx * 2 + 1] = (short)q_acc;
//	}
//}
//
//extern "C" void handleData_in_kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	
//	size_t sizeChannel = count * sizeof(channel_t);
//	size_t sizeIq_buff = iq_buff_size * 2 * sizeof(short);
//	size_t sizeGain = count * sizeof(int);
//	
//	int *dev_gain;
//	short *dev_iq_buff;
//	channel_t *dev_chan;
//
//	int blockSize = 1024;
//	int gridSize = (iq_buff_size + blockSize - 1) / blockSize;
//
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start);
//
//	checkCuda(cudaMalloc((void**)&dev_chan, sizeChannel));
//	checkCuda(cudaMalloc((void**)&dev_gain, sizeGain));
//	checkCuda(cudaMalloc((void**)&dev_iq_buff, sizeIq_buff));
//
//	checkCuda(cudaMemcpy(dev_gain, gain, sizeGain, cudaMemcpyHostToDevice));
//	checkCuda(cudaMemcpy(dev_chan, chan, sizeChannel, cudaMemcpyHostToDevice));
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);
//
//	cudaEventRecord(start);
//	kernel << < gridSize, blockSize>> > (dev_chan, dev_gain, delt, count, iq_buff_size, dev_iq_buff);
//	cudaThreadSynchronize();
//
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds2 = 0;
//	cudaEventElapsedTime(&milliseconds2, start, stop);
//	printf("\nthoi tian thuc hien trong gpu: %f", milliseconds2);
//	checkCuda(cudaMemcpy(iq_buff, dev_iq_buff, sizeIq_buff, cudaMemcpyDeviceToHost));
//
//	checkCuda(cudaFree(dev_chan));
//	checkCuda(cudaFree(dev_gain));
//	checkCuda(cudaFree(dev_iq_buff));
//}