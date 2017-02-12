#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define LIMIT(a,b,c) (MIN(MAX((a),(b)),(c)))

#define RED		0
#define GREEN	1
#define BLUE	2

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
			    }														\
} while (0)

__global__ void kernel_main(int height, int width, int r, unsigned int *src, unsigned int *dst)
{
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
	while (tid_y < height)
	{
		int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
		while (tid_x < width)
		{
			int arr_size;
			arr_size = (MIN(tid_y + r, height - 1) - MAX(tid_y - r, 0) + 1) * (MIN(tid_x + r, width - 1) - MAX(tid_x - r, 0) + 1);

			int median = arr_size / 2 + 1;

			unsigned int answer = 0;

			for (int j = 0; j < 3; j++)
			{
				unsigned short int C[256];
				memset(&C, 0, sizeof(unsigned short int) * 256);

				int y_up = MAX(tid_y - r, 0);
				int y_down = MIN(tid_y + r, height - 1);

				int x_left = MAX(tid_x - r, 0);
				int x_right = MIN(tid_x + r, width - 1);

				for (int k = y_up; k <= y_down; k++)
				{
					for (int l = x_left; l <= x_right; l++)
					{
						C[(src[k * width + l] >> (8 * j)) & 0xFF]++;
					}
				}

				if (C[0] < median)
				{
					for (int i = 1; i < 256; i++)
					{
						C[i] += C[i - 1];
						if (C[i] >= median)
						{
							answer |= i << (8 * j);
							break;
						}
					}
				}
			}
			dst[tid_y * width + tid_x] = answer;
			tid_x += blockDim.x * gridDim.x;
		}
		tid_y += blockDim.y * gridDim.y;
	}
}

int main()
{
	string path_in, path_out;

	cin >> path_in >> path_out;

	int r;
	cin >> r;

	int width, height;
	FILE *in = fopen(path_in.c_str(), "rb");
	if (in == NULL)
	{
		cout << "ERROR: Incorrect input file.\n";
		return 0;
	}
	fread(&width, sizeof(int), 1, in);
	fread(&height, sizeof(int), 1, in);

	if (width <= 0 || height <= 0 || r < 0 || r > 100)
	{
		cout << "ERROR: Incorrect data.\n";
		return 0;
	}

	unsigned int *src = (unsigned int *)malloc(sizeof(unsigned int) * width * height);
	unsigned int *dst = (unsigned int *)malloc(sizeof(unsigned int) * width * height);
	fread(src, sizeof(unsigned int), width * height, in);
	fclose(in);

	unsigned int *src_dev;
	CSC(cudaMalloc(&src_dev, sizeof(unsigned int) * height * width));
	CSC(cudaMemcpy(src_dev, src, sizeof(unsigned int) * height * width, cudaMemcpyHostToDevice));

	free(src);

	unsigned int *dst_dev;
	CSC(cudaMalloc(&dst_dev, sizeof(unsigned int) * height * width));

	dim3 threads_count(16, 16);

	unsigned int blocks_count_x = LIMIT(1, width  / threads_count.x + 1, 32);
	unsigned int blocks_count_y = LIMIT(1, height / threads_count.y + 1, 32);
	dim3 blocks_count(blocks_count_x, blocks_count_y);

	/*cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/

	kernel_main << < blocks_count, threads_count >> >(height, width, r, src_dev, dst_dev);

	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);*/

	CSC(cudaFree(src_dev));
	CSC(cudaMemcpy(dst, dst_dev, sizeof(unsigned int) * height * width, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dst_dev));

	FILE *out = fopen(path_out.c_str(), "wb");
	if (out == NULL)
	{
		cout << "ERROR: Incorrect output file.\n";
		return 0;
	}
	fwrite(&width, sizeof(int), 1, out);
	fwrite(&height, sizeof(int), 1, out);
	fwrite(dst, sizeof(unsigned int), height * width, out);
	fclose(out);

	free(dst);

	/*float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << milliseconds << "\n";*/
	//cudaProfilerStop();
	return 0;
}