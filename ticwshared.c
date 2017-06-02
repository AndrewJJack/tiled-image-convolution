#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
	    }                                                                     \
    } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//TODO: INSERT CODE HERE
__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	for (int i = 0; i < (width - 1) / TILE_WIDTH + 1; ++i){
		if (Row < height && (i * TILE_WIDTH + tx) < width){
			Mds[ty][tx] = I[Row * width + (i * TILE_WIDTH + tx)];
		}
		else {
			Mds[ty][tx] = 0;
		}

		if (Row < Mask_width && (i * TILE_WIDTH + tx) < Mask_width){
			Nds[ty][tx] = I[Row * width + (i * TILE_WIDTH + tx)];
		}
		else {
			Nds[ty][tx] = 0;
		}

		__syncthreads();

		int N_start_col = Col - Mask_radius;
		int N_start_row = Row - Mask_radius;

		for (int c = 0; c < channels; c++){
			float pixVal = 0;
			for (int j = 0; j < Mask_width; ++j) {
				for (int k = 0; k < Mask_width; ++k) {
					int curRow = N_start_row + j;
					int curCol = N_start_col + k;
					// Verify we have a valid image pixel
					if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
						pixVal += Mds[ty*channels+c][j] * Nds[j][tx];
					}

				}
			}
			// Write our new pixel value out
			P[(Row * width + Col)*channels + c] = clamp(pixVal);
		}
	}
	





}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//TODO: INSERT CODE HERE
	cudaMalloc((void **)&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth *imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns*sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns*sizeof(float), cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	dim3 dimBlock(16, 16);
	dim3 dimGrid((imageWidth - 1) / 16 + 1, (imageHeight - 1) / 16 + 1);

	convolution << <dimGrid, dimBlock >> >(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}