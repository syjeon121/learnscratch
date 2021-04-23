///function for deep learning calculation
#include "Function.cuh"
cudaError_t cudaStatus;

void Check(int* output_shape, float* dev_out)
{
	int size_result = 1;
	for (int i = 0; i < 4; i++) size_result *= output_shape[i];
	cout << "size: " << size_result << endl;
	float* host_result = new float[size_result];
	memset(host_result, 0, size_result * sizeof(float));
	cudaMemcpy(host_result, dev_out, size_result * sizeof(float), cudaMemcpyDeviceToHost);

	float sum = 0;
	for (int i = 0; i < size_result; i++) sum += host_result[i];

	cout << "average of output: " << sum / (float)size_result << endl;


	int N = output_shape[0]; cout << "N=" << N << endl;
	int C = output_shape[1]; cout << "C=" << C << endl;
	int H = output_shape[2]; cout << "H=" << H << endl;
	int W = output_shape[3]; cout << "W=" << W << endl;



	float sum1 = 0;
	float tmp1;
	for (int l = 0; l < W; l++) {

		tmp1 = host_result[0 * (C*H*W) + 0 * (H*W) + 0 * (W)+l];
		cout << "value of output[0,0,0,:]: " << tmp1 << endl;
		sum1 += tmp1;
	}
	cout << "average of output[0,0,0,:]: " << sum1 / (float)W << endl;





	delete[] host_result;
}

//index change
__device__ __host__ void idx2d(int tid,
	int ni, int nj,
	int& i, int& j) {

	i = tid / nj;
	j = tid % nj;
}

__device__ __host__ void idx4d(int tid,
	int ni, int nj, int nk, int nl,
	int& i, int& j, int& k, int& l) {

	i = tid / (nj*nk*nl);
	tid = tid - (i*(nj*nk*nl));

	j = tid / (nk*nl);
	tid = tid - (j*(nk*nl));

	k = tid / (nl);
	l = tid % (nl);

}

__device__ __host__ void idx6d(int tid,
	int ni, int nj, int nk, int nl, int nm, int nn,
	int& i, int& j, int& k, int& l, int& m, int& n) {

	i = tid / (nj*nk*nl*nm*nn);
	tid = tid - (i*(nj*nk*nl*nm*nn));

	j = tid / (nk*nl*nm*nn);
	tid = tid - (j*(nk*nl*nm*nn));

	k = tid / (nl*nm*nn);
	tid = tid - (k*(nl*nm*nn));

	l = tid / (nm*nn);
	tid = tid - (l*(nm*nn));

	m = tid / (nn);
	n = tid % (nn);
}










void Dot(float* C, float* A, float* B, const int r, const int c, const int n) {

	float temp;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			temp = 0.0;
			for (int k = 0; k < n; k++) {
				temp += A[i*n + k] * B[k*c + j];
			}
			C[i*c + j] = temp;
		}
	}

}

__global__ void Kernel_Dot(float* C, float* A, float* B,
	const int r, const int c, const int n) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int N = r*c;
	int i, j;
	float temp, A_val, B_val;

	while (tid < N)
	{
		temp = 0.0;
		A_val = 0.0;
		B_val = 0.0;

		idx2d(tid, r, c, i, j);

		for (int k = 0; k < n; k++) {
			A_val = A[i*n + k];
			B_val = B[k*c + j];
			temp += A_val*B_val;
		}

		C[i*c + j] = temp;

		tid += gridDim.x*blockDim.x;
	}

}

void Dot_gpu(float* dev_C, float* dev_A, float* dev_B,
	const int r, const int c, const int n) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Dot << < dimGrid, dimBlock >> > (dev_C, dev_A, dev_B, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Dot_coalescing1(float* C, float* A, float* B,
	const int r, const int c, const int n) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int N = r*c;
	int i, j;
	float temp, A_val, B_val;

	while (tid < N)
	{
		temp = 0.0;
		A_val = 0.0;
		B_val = 0.0;

		idx2d(tid, r, c, i, j);

		for (int k = 0; k < n; k++) {
			A_val = A[k*r + i];
			B_val = B[k*c + j];
			temp += A_val*B_val;
		}

		C[i*c + j] = temp;

		tid += gridDim.x*blockDim.x;
	}

}

void Dot_coalescing1_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Dot_coalescing1 << < dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Dot_coalescing2(float* C, float* A, float* B,
	const int r, const int c, const int n) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int N = r*c;
	int i, j;
	float temp, A_val, B_val;

	while (tid < N)
	{
		temp = 0.0;
		A_val = 0.0;
		B_val = 0.0;

		idx2d(tid, r, c, i, j);

		for (int k = 0; k < n; k++) {
			A_val = A[i*n + k];
			B_val = B[j*n + k];
			temp += A_val*B_val;
		}

		C[i*c + j] = temp;

		tid += gridDim.x*blockDim.x;
	}

}

void Dot_coalescing2_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Dot_coalescing2 << < dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Dot_reduction1(float* dev_a, float* dev_b,
	const int r, const int c, const int n,
	float* reduction) {

	__shared__ float shared[BLOCK_SIZE];

	unsigned int k = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int sharedIdx = threadIdx.x;
	if (k >= n) return;

	float A_val = 0;
	float B_val = 0;

	int m;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {

			A_val = dev_a[i*n + k];
			B_val = dev_b[k*c + j];

			shared[sharedIdx] = A_val*B_val;
			__syncthreads();


			m = blockDim.x / 2;
			while (m != 0) {
				if (sharedIdx < m) shared[sharedIdx] += shared[sharedIdx + m];
				__syncthreads();
				m /= 2;
			}


			if (sharedIdx == 0) reduction[i*(c*gridDim.x) + j*(gridDim.x) + blockIdx.x] = shared[0];
			__syncthreads();
		}
	}




}

__global__ void Kernel_Dot_reduction2(float* dev_c, float* reduction, int r, const int c, const int n,
	int size_block) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= r || j >= c) return;

	float temp = 0;
	for (int k = 0; k < size_block; k++) {
		temp += reduction[i*(c*size_block) + j*(size_block)+k];
	}


	dev_c[i*c + j] = temp;

}

void Dot_reduction_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n,
	float* reduction)
{
	dim3 dimBlock1(BLOCK_SIZE);
	dim3 dimGrid1((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	Kernel_Dot_reduction1 << < dimGrid1, dimBlock1 >> > (dev_a, dev_b, r, c, n, reduction);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	int size_block = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimBlock2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid2((r + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (c + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	Kernel_Dot_reduction2 << < dimGrid2, dimBlock2 >> > (dev_c, reduction, r, c, n, size_block);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Dot_atomic(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n) {

	unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int N = n;
	float temp, A_val, B_val;

	while (k < N)
	{
		for (int i = 0; i < r; i++)
		{
			for (int j = 0; j < c; j++)
			{
				A_val = dev_a[i*n + k];
				B_val = dev_b[k*c + j];
				temp = A_val * B_val;
				atomicAdd(&(dev_c[i*c + j]), temp);
			}
		}


		k += gridDim.x*blockDim.x;
	}

}

void Dot_atomic_gpu(float* dev_C, float* dev_A, float* dev_B,
	const int r, const int c, const int n) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Dot_atomic << < dimGrid, dimBlock >> > (dev_C, dev_A, dev_B, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

void Sum(char txt, float* A, float* B, const int r, const int c) {

	switch (txt)
	{
	case 'f':

		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				A[i*c + j] += B[j];
			}
		}
		break;
	case 'b':

		for (int j = 0; j < c; j++) {
			A[j] = 0.0;
		}

		for (int j = 0; j < c; j++) {
			for (int i = 0; i < r; i++) {
				A[j] += B[i*c + j];
			}
		}

		break;
	default:
		cout << "Error for 'txt' variable!" << endl;
		break;
	}


}

__global__ void Kernel_Sum_forward(float* dev_A, float* dev_B, const int r, const int c) {



	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r*c;
	int i, j;

	while (tid < N)
	{
		idx2d(tid, r, c, i, j);

		dev_A[i*c + j] += dev_B[j];

		tid += gridDim.x*blockDim.x;
	}

}

__global__ void Kernel_Sum_backward(float* dev_A, float* dev_B, const int r, const int c) {


	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = c;
	int j;

	while (tid < N)
	{
		j = tid;

		dev_A[j] = 0.0;

		for (int i = 0; i < r; i++) {
			dev_A[j] += dev_B[i*c + j];
		}

		tid += gridDim.x*blockDim.x;
	}


}

//template <unsigned int blockSize>
//__device__ void warpReduce(volatile float* sdata, int tid)
//{
//	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//}

//__global__ void Kernel_Sum_backward_opt(float* dev_sum, float* dev_B, const int r, const int c) {


//	__shared__ float sdata[(BLOCK_SIZE_opt / 2)];

//	unsigned int tid = threadIdx.x;
//	unsigned int i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
//	//if (i >= r) return;

//	for (int j = 0; j < c; j++) {

//		sdata[tid] = dev_B[i*c + j] + dev_B[(i + blockDim.x)*c + j];
//		__syncthreads();

//		if (blockDim.x >= 512) {
//			if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
//		}
//		if (blockDim.x >= 256) {
//			if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
//		}
//		if (blockDim.x >= 128) {
//			if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
//		}

//		if (tid < 32) warpReduce<BLOCK_SIZE_opt / 2>(sdata, tid);

//		if (tid == 0) dev_sum[blockIdx.x*c + j] = sdata[0];
//		__syncthreads();

//	}
//}

__global__ void Kernel_Sum_backward_opt_sum(float* dev_A, float* dev_sum, int r_sum, const int c) {

	unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= c) return;

	float temp = 0;
	for (int i = 0; i < r_sum; i++) {
		temp += dev_sum[i*c + j];
	}

	dev_A[j] = temp;

}

__global__ void Kernel_Sum_backward1(float* dev_B, float* dev_partial, const int r, const int c) {


	__shared__ float cache[BLOCK_SIZE];

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int cacheIndex = threadIdx.x;
	if (i >= r) return;

	for (int j = 0; j < c; j++) {
		cache[cacheIndex] = dev_B[i*c + j];
		__syncthreads();


		int k = blockDim.x / 2;
		while (k != 0) {
			if (cacheIndex < k) cache[cacheIndex] += cache[cacheIndex + k];
			__syncthreads();
			k /= 2;
		}

		if (cacheIndex == 0) dev_partial[blockIdx.x*c + j] = cache[0];
		__syncthreads();

	}
}

__global__ void Kernel_Sum_backward2(float* dev_A, float* dev_partial, const int r, const int c,
	int size_partial) {

	unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= c) return;

	int i;
	float temp = 0;
	for (i = 0; i < size_partial; i++) {
		temp += dev_partial[i*c + j];
	}

	dev_A[j] = temp;

}


void Sum_gpu(char txt, float* dev_A, float* dev_B, const int r, const int c) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	switch (txt)
	{
	case 'f':
		Kernel_Sum_forward << < dimGrid, dimBlock >> > (dev_A, dev_B, r, c);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		break;
	case 'b':
		Kernel_Sum_backward << < dimGrid, dimBlock >> > (dev_A, dev_B, r, c);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
		break;
	default:
		cout << "Error for 'txt' variable!" << endl;
		break;
	}
}

void Sum_gpu(char txt, float* dev_A, float* dev_B, const int r, const int c,
	float* dev_sum)
{
	if (txt != 'b')
		cout << "(Sum_gpu) this function should be in backward" << endl;

	dim3 dimBlock(BLOCK_SIZE_opt / 2);		//halve the number of threads
	dim3 dimGrid((r + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt);
	//Kernel_Sum_backward_opt << < dimGrid, dimBlock >> > (dev_sum, dev_B, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	int r_sum = (r + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt;
	dim3 dimBlock_sum(BLOCK_SIZE_opt);
	dim3 dimGrid_sum((c + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt);
	Kernel_Sum_backward_opt_sum << < dimGrid_sum, dimBlock_sum >> > (dev_A, dev_sum, r_sum, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Sum_gpu1(char txt, float* dev_A, float* dev_B, const int r, const int c,
	float* dev_partial, int size_partial) {

	dim3 dimBlock2(BLOCK_SIZE);
	dim3 dimGrid2((r + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid2.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Sum_gpu'!" << endl;
	}

	Kernel_Sum_backward1 << < dimGrid2, dimBlock2 >> > (dev_B, dev_partial, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	dim3 dimBlock3(BLOCK_SIZE);
	dim3 dimGrid3((c + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid3.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Sum_gpu'!" << endl;
	}


	Kernel_Sum_backward2 << < dimGrid3, dimBlock3 >> > (dev_A, dev_partial, r, c, size_partial);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

/*loss function*/

float MSE(float** x1, float** x2, const int r, const int c) {

	float temp = 0.0;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			temp += pow(x1[i][j] - x2[i][j], 2);
		}
	}

	temp /= 2.0*r;
	return temp;

}

void Softmax(float* x, const int r, const int c) {

	float temp1, temp2;
	for (int i = 0; i < r; i++) {
		temp1 = 0.;
		temp2 = 0.;

		for (int j = 0; j < c; j++)
		{
			temp1 = max(x[i*c + j], temp1);
		}

		for (int j = 0; j < c; j++)
		{
			x[i*c + j] = expf(x[i*c + j] - temp1);
			temp2 += x[i*c + j];
		}

		for (int j = 0; j < c; j++) x[i*c + j] /= temp2;
	}
}

void Softmax_seg(float* x, const int size_category, const int size_spatial_feature_map)
{
	int c = size_category;
	int size = size_spatial_feature_map;

	float temp1, temp2;
	for (int i = 0; i < size; i++) {
		temp1 = 0.;
		temp2 = 0.;

		for (int j = 0; j < c; j++)
		{
			temp1 = max(x[j*size + i], temp1);
		}

		for (int j = 0; j < c; j++)
		{
			x[j*size + i] = expf(x[j*size + i] - temp1);
			temp2 += x[j*size + i];
		}

		for (int j = 0; j < c; j++) x[j*size + i] /= temp2;
	}
}

__global__ void Kernel_Softmax(float* dev_x, const int r, const int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= r) return;

	float temp1 = 0., temp2 = 0.;
	for (int j = 0; j < c; j++) temp1 = max(dev_x[i*c + j], temp1);

	for (int j = 0; j < c; j++) {
		dev_x[i*c + j] = expf(dev_x[i*c + j] - temp1);
		temp2 += dev_x[i*c + j];
	}


	for (int j = 0; j < c; j++) dev_x[i*c + j] /= temp2;
}

void Softmax_gpu(float* dev_x, const int r, const int c) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((r + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax_gpu'!" << endl;
	}

	Kernel_Softmax << < dimGrid, dimBlock >> > (dev_x, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Softmax_seg(float* dev_x, const int c, const int size) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = size;
	float temp = 0.;

	while (i < N)
	{
		for (int j = 0; j < c; j++)
			temp = max(dev_x[j*size + i], temp);

		for (int j = 0; j < c; j++)
			dev_x[j*size + i] = expf(dev_x[j*size + i] - temp);

		temp = 0.0;
		for (int j = 0; j < c; j++)
			temp += dev_x[j*size + i];

		for (int j = 0; j < c; j++)
			dev_x[j*size + i] /= temp;


		i += gridDim.x*blockDim.x;
	}

}

void Softmax_seg_gpu(float* dev_x, const int size_category, const int size_spatial_feature_map) {

	int size = size_spatial_feature_map;
	int c = size_category;

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Softmax_seg << < dimGrid, dimBlock >> > (dev_x, c, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Softmax4d(float* dev_x, int N, int C, int H, int W) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int Max = H*W;
	if (tid >= Max) return;

	int i, j;
	idx2d(tid, H, W, i, j);

	int idx;


	float temp_max = 0;
	for (int n = 0; n < C; n++) temp_max = max(dev_x[0 * (C*H*W) + n*(H*W) + i*(W)+j], temp_max);

	float temp_sum = 0;
	for (int n = 0; n < C; n++) temp_sum += expf(dev_x[0 * (C*H*W) + n*(H*W) + i*(W)+j] - temp_max);

	for (int n = 0; n < C; n++) dev_x[0 * (C*H*W) + n*(H*W) + i*(W)+j] = expf(dev_x[0 * (C*H*W) + n*(H*W) + i*(W)+j] - temp_max) / temp_sum;


}

void Softmax4d_gpu(float* dev_x, int N, int C, int H, int W) {

	if (N != 1) // the batch size 'XN' must be 1
	{
		cout << "the batch size 'N' must be 1! N=[" << N << "] at Softmax4d_gpu" << endl;
	}

	int size = H*W;
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax4d_gpu'!" << endl;
	}

	Kernel_Softmax4d << < dimGrid, dimBlock >> > (dev_x, N, C, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Softmax_shared1(float* dev_x, const int XN, const int DN, float* dev_partialX4d, int size_partialX4d) {

	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= DN) return;

	cache[cacheIdx] = dev_x[tid];
	__syncthreads();


	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] = max(cache[cacheIdx], cache[cacheIdx + k]);
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_partialX4d[blockIdx.x] = cache[0];
	__syncthreads();


}

__global__ void Softmax_shared2(float* dev_x, const int XN, const int DN, float* dev_partialX4d, int size_partialX4d) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= DN) return;

	float sum = 0;
	for (int i = 0; i < size_partialX4d; i++)
	{
		sum += dev_partialX4d[i];
	}


	dev_x[tid] = expf(dev_x[tid] - sum);



	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();

	cache[cacheIdx] = dev_x[tid];
	__syncthreads();


	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_partialX4d[blockIdx.x] = cache[0];
	__syncthreads();

}

__global__ void Softmax_shared3(float* dev_x, const int XN, const int DN, float* dev_partialX4d, int size_partialX4d) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= DN) return;

	float sum = 0;
	for (int i = 0; i < size_partialX4d; i++)
	{
		sum += dev_partialX4d[i];
	}


	dev_x[tid] /= sum;

}

void Softmax_gpu_shared(float* dev_x, const int XN, const int DN, float* dev_partialX4d, int size_partialX4d) {

	if (XN != 1) // the batch size 'XN' must be 1
	{
		cout << "the batch size 'XN' must be 1! XN=[" << XN << ']' << endl;
	}

	dim3 dimBlock1(BLOCK_SIZE);
	dim3 dimGrid1((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid1.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax_gpu_shared1'!" << endl;
	}
	Softmax_shared1 << < dimGrid1, dimBlock1 >> > (dev_x, XN, DN, dev_partialX4d, size_partialX4d);
	gpuErrchk(cudaGetLastError());


	dim3 dimBlock2(BLOCK_SIZE);
	dim3 dimGrid2((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid2.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax_gpu_shared2'!" << endl;
	}
	Softmax_shared2 << < dimGrid2, dimBlock2 >> > (dev_x, XN, DN, dev_partialX4d, size_partialX4d);
	gpuErrchk(cudaGetLastError());


	dim3 dimBlock3(BLOCK_SIZE);
	dim3 dimGrid3((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid3.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax_gpu_shared3'!" << endl;
	}
	Softmax_shared3 << < dimGrid3, dimBlock3 >> > (dev_x, XN, DN, dev_partialX4d, size_partialX4d);
	gpuErrchk(cudaGetLastError());

}

float CEE_seg(float* x, int* t, const int size_category, const int size_spatial_feature_map)
{
	int c = size_category;
	int size = size_spatial_feature_map;


	float temp = 0;
	for (int j = 0; j < size; j++) {
		for (int i = 0; i < c; i++) {

			if (i == t[j]) {
				temp += log(x[i*size + j] + 1e-7);
				continue;
			}
		}
	}

	temp /= -size;
	return temp;
}

__global__ void Kernel_CEE_seg(float* dev_x, int* dev_t, float* dev_loss, const int c, const int size)
{
	int j = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	float temp = 0;
	while (j < N)
	{
		for (int i = 0; i < c; i++) {

			if (i == dev_t[j]) {
				temp = logf(dev_x[i*size + j] + 1e-7);
				atomicAdd(dev_loss, temp);
				continue;
			}
		}

		j += gridDim.x*blockDim.x;
	}
}

float CEE_seg_gpu(float* dev_x, int* dev_t, float* dev_loss,
	const int size_category, const int size_spatial_feature_map) {

	int c = size_category;
	int size = size_spatial_feature_map;

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	cudaMemset(dev_loss, 0, sizeof(float));
	Kernel_CEE_seg << < dimGrid, dimBlock >> > (dev_x, dev_t, dev_loss, c, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	float loss = 0;
	cudaMemcpy(&loss, dev_loss, sizeof(float), cudaMemcpyDeviceToHost);
	loss /= -size;

	return loss;

}




/*padding and stride*/

void Padding_forward(char txt, float* x_pad, float* x, const int pad,
	const int XN, const int XC, const int XH, const int XW) {


	int idx, idx_pad;
	int XH_pad = XH + 2 * pad;
	int XW_pad = XW + 2 * pad;

	for (int i = 0; i < XN; i++) {
		for (int j = 0; j < XC; j++) {
			for (int k = 0; k < XH; k++) {
				for (int l = 0; l < XW; l++) {

					idx = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
					idx_pad = i*(XC*XH_pad*XW_pad) + j*(XH_pad*XW_pad) + (k + pad)*(XW_pad)+(l + pad);

					x_pad[idx_pad] = x[idx];

				}
			}
		}
	}

}

void Padding_backward(char txt, float* dx_pad, float* dx, const int pad,
	const int XN, const int XC, const int XH, const int XW,
	const int dXH, const int dXW)
{
	int i, j, k, l;
	int idx_dx, idx_dx_pad;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < XH; k++) {
				for (l = 0; l < XW; l++) {

					idx_dx_pad = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
					idx_dx = i*(XC*dXH*dXW) + j*(dXH*dXW) + (k + pad)*(dXW)+(l + pad);

					dx_pad[idx_dx_pad] = dx[idx_dx];
				}
			}
		}
	}

}

__global__ void Kernel_Padding_forward(float* dev_x_pad, float*dev_X, const int pad,
	const int XN, const int XC, const int XH, const int XW) {


	int i, j, k, l;

	int XH_pad = XH + 2 * pad;
	int XW_pad = XW + 2 * pad;

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;

	int idx_pad, idx;

	while (tid < N)
	{
		idx4d(tid, XN, XC, XH, XW, i, j, k, l);
		idx_pad = i*(XC*XH_pad*XW_pad) + j*(XH_pad*XW_pad) + (k + pad)*(XW_pad)+(l + pad);
		idx = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;

		dev_x_pad[idx_pad] = dev_X[idx];

		tid += gridDim.x*blockDim.x;
	}



}

void Padding_forward_gpu(float* dev_x_pad, float* dev_X, const int pad,
	const int XN, const int XC, const int XH, const int XW) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Padding_forward << < dimGrid, dimBlock >> > (dev_x_pad, dev_X, pad, XN, XC, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Padding_backward(float* dev_dx_pad, float*dev_dx, const int pad,
	const int XN, const int XC, const int XH, const int XW,
	const int dXH, const int dXW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	int i, j, k, l, idx_dx_pad, idx_dx;
	while (tid < N)
	{
		idx4d(tid, XN, XC, XH, XW, i, j, k, l);
		idx_dx_pad = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
		idx_dx = i*(XC*dXH*dXW) + j*(dXH*dXW) + (k + pad)*(dXW)+(l + pad);

		dev_dx_pad[idx_dx_pad] = dev_dx[idx_dx];

		tid += gridDim.x*blockDim.x;
	}

}

void Padding_backward_gpu(float* dev_dx_pad, float*dev_dx, const int pad,
	const int XN, const int XC, const int XH, const int XW,
	const int dxH, const int dxW) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Padding_backward << < dimGrid, dimBlock >> > (dev_dx_pad, dev_dx, pad, XN, XC, XH, XW, dxH, dxW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Padding_transpose_forward(float* x_pad, float* x, int stride, int pad,
	int XN, int XC, int XH, int XW, int XH_pad, int XW_pad)
{
	int idx_pad, idx;
	for (int i = 0; i < XN; i++) {
		for (int j = 0; j < XC; j++) {
			for (int k = 0; k < XH; k++) {
				for (int l = 0; l < XW; l++) {

					idx_pad = i*(XC*XH_pad*XW_pad) + j*(XH_pad*XW_pad) + (stride*k + pad)*(XW_pad)+(stride*l + pad);
					idx = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;

					x_pad[idx_pad] = x[idx];

				}
			}
		}
	}
}

void Padding_transpose_backward(float* dx_pad, float* dx, int stride, int pad,
	int XN, int XC, int XH, int XW, int dXH, int dXW)
{
	int idx_dx_pad, idx_dx;
	for (int i = 0; i < XN; i++) {
		for (int j = 0; j < XC; j++) {
			for (int k = 0; k < XH; k++) {
				for (int l = 0; l < XW; l++) {

					idx_dx_pad = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
					idx_dx = i*(XC*dXH*dXW) + j*(dXH*dXW) + (stride*k + pad)*(dXW)+(stride*l + pad);

					dx_pad[idx_dx_pad] = dx[idx_dx];
				}
			}
		}
	}
}

__global__ void Kernel_Padding_transpose_forward(float* dev_x_pad, float* dev_x, int stride, int pad,
	int XN, int XC, int XH, int XW, int XH_pad, int XW_pad) {


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	int i, j, k, l, idx_x_pad, idx_x;

	while (tid < N)
	{
		idx4d(tid, XN, XC, XH, XW, i, j, k, l);
		idx_x_pad = i*(XC*XH_pad*XW_pad) + j*(XH_pad*XW_pad) + (stride*k + pad)*(XW_pad)+(stride*l + pad);
		idx_x = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;

		dev_x_pad[idx_x_pad] = dev_x[idx_x];

		tid += gridDim.x*blockDim.x;
	}


}

void Padding_transpose_forward_gpu(float* dev_x_pad, float* dev_x, int stride, int pad,
	int XN, int XC, int XH, int XW, int XH_pad, int XW_pad) {



	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);


	Kernel_Padding_transpose_forward << < dimGrid, dimBlock >> > (dev_x_pad, dev_x, stride, pad, XN, XC, XH, XW, XH_pad, XW_pad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());



}

__global__ void Kernel_Padding_transpose_backward(float* dev_dx_pad, float* dev_dx, int stride, int pad,
	int XN, int XC, int XH, int XW, int dXH, int dXW)
{
	int i, j, k, l;

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	if (tid >= N) return;

	idx4d(tid, XN, XC, XH, XW, i, j, k, l);


	int idx_dx_pad = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
	int idx_dx = i*(XC*dXH*dXW) + j*(dXH*dXW) + (stride*k + pad)*(dXW)+(stride*l + pad);

	dev_dx_pad[idx_dx_pad] = dev_dx[idx_dx];

}

void Padding_transpose_backward_gpu(float* dev_dx_pad, float* dev_dx, int stride, int pad,
	int XN, int XC, int XH, int XW, int dXH, int dXW)
{
	int size = XN*XC*XH*XW;
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'padding_transpose_backward_gpu'!" << endl;
	}
	Kernel_Padding_transpose_backward << < dimGrid, dimBlock >> > (dev_dx_pad, dev_dx, stride, pad, XN, XC, XH, XW, dXH, dXW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

void Stride_forward(float* col, float* img, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {


	int i, j, k, l, m, n, a, b;
	int y_max, x_max;
	int idx_col, idx_img;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < FH; k++) {
				for (l = 0; l < FW; l++) {
					for (m = 0; m < OH; m++) {
						for (n = 0; n < OW; n++) {

							idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
							col[idx_col] = 0;
						}
					}
				}
			}
		}
	}




	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {

			for (k = 0; k < FH; k++) {
				y_max = k + stride*OH;

				for (l = 0; l < FW; l++) {
					x_max = l + stride*OW;

					for (a = k, m = 0; a < y_max; a = a + stride, m++) {
						for (b = l, n = 0; b < x_max; b = b + stride, n++) {

							idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
							idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;

							col[idx_col] = img[idx_img];

						}
					}
				}
			}
		}
	}
}

__global__ void Kernel_Stride_forward(float* dev_col, float* dev_img, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*FH*FW*OH*OW;
	int i, j, k, l, m, n, a, b;

	int idx_col;
	int idx_img;

	while (tid < N)
	{
		idx6d(tid, XN, XC, FH, FW, OH, OW, i, j, k, l, m, n);

		a = k + m*stride;
		b = l + n*stride;

		idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
		idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;

		dev_col[idx_col] = 0;
		dev_col[idx_col] = dev_img[idx_img];

		tid += gridDim.x * blockDim.x;
	}

}

void Stride_forward_gpu(float* dev_col, float* dev_img, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {



	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Stride_forward << < dimGrid, dimBlock >> > (dev_col, dev_img, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

void Stride_backward(float* img, float* col, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {


	int i, j, k, l, m, n;
	int y_max, x_max;
	int idx_img, idx_col;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < XH; k++) {
				for (l = 0; l < XW; l++) {

					idx_img = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
					img[idx_img] = 0;
				}
			}
		}
	}

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {

			for (k = 0; k < FH; k++) {
				y_max = k + stride*OH;

				for (l = 0; l < FW; l++) {
					x_max = l + stride*OW;


					for (int a = k, m = 0; a < y_max; a = a + stride, m++) {

						for (int b = l, n = 0; b < x_max; b = b + stride, n++) {

							idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;
							idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;


							img[idx_img] += col[idx_col];

						}
					}
				}
			}
		}
	}
}

__global__ void Kernel_Stride_backward(float* dev_img, float* dev_col, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	int i, j, a, b, idx_img, idx_col;
	int k, l, m, n, temp;

	while (tid < N)
	{
		idx4d(tid, XN, XC, XH, XW, i, j, a, b);

		idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;
		dev_img[idx_img] = 0;


		for (k = 0; k < FH && k <= a; k++)
		{
			m = (a - k) / stride;
			temp = k + stride*m;
			if (temp != a || m >= OH)
				continue;


			for (l = 0; l < FW && l <= b; l++)
			{
				n = (b - l) / stride;
				temp = l + stride*n;
				if (temp != b || n >= OW)
					continue;


				idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;

				dev_img[idx_img] += dev_col[idx_col];

			}
		}



		tid += gridDim.x*blockDim.x;
	}

}

void Stride_backward_gpu(float* dev_img, float* dev_col, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Stride_backward << < dimGrid, dimBlock >> > (dev_img, dev_col, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}



/*reshape and transpose*/

void Flatten6d(float* flattenX, float****** X,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6) {

	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d2; j++) {
			for (int k = 0; k < d3; k++) {
				for (int l = 0; l < d4; l++) {
					for (int m = 0; m < d5; m++) {
						for (int n = 0; n < d6; n++) {
							flattenX[i*(d2*d3*d4*d5*d6) + j*(d3*d4*d5*d6) + k*(d4*d5*d6) + l*(d5*d6) + m*(d6)+n]
								= X[i][j][k][l][m][n];
						}
					}
				}
			}
		}
	}
}

void Flatten4d(float* flattenX, float**** X,
	const int d1, const int d2, const int d3, const int d4) {

	int i, j, k, l;

	for (i = 0; i < d1; i++) {
		for (j = 0; j < d2; j++) {
			for (k = 0; k < d3; k++) {
				for (l = 0; l < d4; l++) {
					flattenX[i*(d2*d3*d4) + j*(d3*d4) + k*(d4)+l] = X[i][j][k][l];
				}
			}
		}
	}
}

void Flatten2d(float* flattenX, float** X,
	const int d1, const int d2) {

	int i, j;

	for (i = 0; i < d1; i++) {
		for (j = 0; j < d2; j++) {
			flattenX[i*(d2)+j] = X[i][j];
		}
	}
}

void Flatten2d_int(int* flattenX, int** X,
	const int d1, const int d2) {

	int i, j;

	for (i = 0; i < d1; i++) {
		for (j = 0; j < d2; j++) {
			flattenX[i*(d2)+j] = X[i][j];
		}
	}
}

void Reshape6to2(float** reshapeArray, float****** array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {

	int i, j, k, l, m, n;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					for (m = 0; m < FH; m++) {
						for (n = 0; n < FW; n++) {
							reshapeArray[i*(OH*OW) + j*(OW)+k][l*(FH*FW) + m*(FH)+n] = array[i][j][k][l][m][n];
						}
					}
				}
			}
		}
	}

}

void Reshape6to2_gpu(float* dev_reshapeArray, float* dev_array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW,
	float* host_reshapeArray, int size_reshapeArray) {

	//Kernel_Reshape6to2 << < 1, 1 >> > (dev_reshapeArray, dev_array, XN, OH, OW, XC, FH, FW);
	//cudaDeviceSynchronize();
	//cudaMemcpy(host_reshapeArray, dev_reshapeArray, size_reshapeArray * sizeof(float), cudaMemcpyDeviceToHost);
}

void Reshape6to2_poolingForward(float** reshapeArray, float****** array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {

	int i, j, k, l, m, n;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					for (m = 0; m < FH; m++) {
						for (n = 0; n < FW; n++) {
							reshapeArray[i*(OH*OW*XC) + j*(OW*XC) + k*(XC)+l][m*(FW)+n] = array[i][j][k][l][m][n];
						}
					}
				}
			}
		}
	}

}

void Reshape4to2_forward(float** reshapeArray, float**** array,
	const int FN, const int FC, const int FH, const int FW) {

	int i, j, k, l;

	for (i = 0; i < FN; i++) {
		for (j = 0; j < FC; j++) {
			for (k = 0; k < FH; k++) {
				for (l = 0; l < FW; l++) {

					reshapeArray[i][j*(FH*FW) + k*(FW)+l] = array[i][j][k][l];

				}
			}
		}
	}

}

void Reshape4to2_backward(float** reshapeArray, float**** array,
	const int XN, const int OH, const int OW, const int FN) {

	int i, j, k, l;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < FN; l++) {

					reshapeArray[i*(OH*OW) + j*(OW)+k][l] = array[i][j][k][l];

				}
			}
		}
	}
}

void Reshape4to2(char txt, float** reshapeArray, float**** array,
	const int d1, const int d2, const int d3, const int d4) {

	int FN, FC, FH, FW, XN, OH, OW;
	int i, j, k, l;

	switch (txt)
	{
	case 'f':
		FN = d1;
		FC = d2;
		FH = d3;
		FW = d4;
		Reshape4to2_forward(reshapeArray, array, FN, FC, FH, FW);

		break;
	case 'b':
		XN = d1;
		OH = d2;
		OW = d3;
		FN = d4;
		Reshape4to2_backward(reshapeArray, array, XN, OH, OW, FN);

		break;
	default:
		cout << "Error for 'txt' variable in Reshape4to2(cpu)!" << endl;
		break;
	}


}

void Reshape2to4_forward(float**** reshapeArray, float** array,
	const int XN, const int OH, const int OW, const int FN) {

	int i, j, k, l;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < FN; l++) {
					reshapeArray[i][j][k][l] = array[i*(OH*OW) + j*(OW)+k][l];
				}
			}
		}
	}

}

void Reshape2to4_backward(float**** reshapeArray, float** array,
	const int XN, const int XC, const int XH, const int XW) {

	int i, j, k, l;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < XH; k++) {
				for (l = 0; l < XW; l++) {
					reshapeArray[i][j][k][l] = array[i][j*(XH*XW) + k*(XW)+l];
				}
			}
		}
	}

}

void Reshape2to4(char txt, float**** reshapeArray, float** array,
	const int d1, const int d2, const int d3, const int d4) {

	int XN, OH, OW, FN, XC, XH, XW;
	int i, j, k, l;

	switch (txt)
	{
	case 'f':
		XN = d1;
		OH = d2;
		OW = d3;
		FN = d4;
		Reshape2to4_forward(reshapeArray, array, XN, OH, OW, FN);

		break;
	case 'b':
		XN = d1;
		XC = d2;
		XH = d3;
		XW = d4;
		Reshape2to4_backward(reshapeArray, array, XN, XC, XH, XW);

		break;
	default:
		cout << "Error for 'txt' variable in Reshape2to4(cpu)!" << endl;
		break;
	}

}

void Reshape2to6(float****** reshapeArray, float** array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {


	int i, j, k, l, m, n;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					for (m = 0; m < FH; m++) {
						for (n = 0; n < FW; n++) {
							reshapeArray[i][j][k][l][m][n] = array[i*(OH*OW) + j*(OW)+k][l*(FH*FW) + m*(FH)+n];
						}
					}
				}
			}
		}
	}




}

void Reshape1to6(float****** reshapeArray, float* array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6) {

	int i, j, k, l, m, n;

	for (i = 0; i < d1; i++) {
		for (j = 0; j < d2; j++) {
			for (k = 0; k < d3; k++) {
				for (l = 0; l < d4; l++) {
					for (m = 0; m < d5; m++) {
						for (n = 0; n < d6; n++) {
							reshapeArray[i][j][k][l][m][n] = array[i*(d2*d3*d4*d5*d6) + j*(d3*d4*d5*d6) + k*(d4*d5*d6) + l*(d5*d6) + m*(d6)+n];
						}
					}
				}
			}
		}
	}

}

void Reshape1to4(float**** reshapeArray, float* array,
	const int XN, const int OH, const int OW, const int XC) {

	int i, j, k, l;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					reshapeArray[i][j][k][l] = array[i*(OH*OW*XC) + j*(OW*XC) + k*(XC)+l];
				}
			}
		}
	}

}

void Reshape1to2(float** reshapeArray, float* array,
	const int d1, const int d2) {

	int i, j;

	for (i = 0; i < d1; i++) {
		for (j = 0; j < d2; j++) {

			reshapeArray[i][j] = array[i*(d2)+j];

		}
	}

}

void Transpose2d(float* array_transpose, float* array, const int r, const int c) {

	int i, j;

	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			array_transpose[j*r + i] = array[i*c + j];
		}
	}

}

__global__ void Kernel_Transpose2d(float* dev_transposeArray, float* dev_array,
	const int r, const int c) {



	//unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//int N = r*c;
	//int i, j, idx_transposeArray, idx_array;

	//while (tid < N)
	//{
	//	idx2d(tid, r, c, i, j);
	//	idx_array = i*c + j;
	//	idx_transposeArray = j*r + i;

	//	dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

	//	tid += gridDim.x * blockDim.x;
	//}


	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= r || j >= c) return;

	int idx_transposeArray, idx_array;

	idx_array = i*c + j;
	idx_transposeArray = j*r + i;

	dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

}

void Transpose2d_gpu(float* dev_transposeArray, float* dev_array, const int r, const int c) {

	//dim3 dimBlock(BLOCK_SIZE);
	//dim3 dimGrid(GRID_SIZE);
	//Kernel_Transpose2d << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, r, c);
	//cudaDeviceSynchronize();
	//gpuErrchk(cudaGetLastError());

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((r + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (c + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	Kernel_Transpose2d << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Transpose4d_forward(float* array_transpose, float* array,
	const int XN, const int OH, const int OW, const int FN) {

	int i, j, k, l;
	int idx_transpose, idx;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < FN; l++) {

					idx_transpose = i*(FN*OH*OW) + l*(OH*OW) + j*(OW)+k;
					idx = i*(OH*OW*FN) + j*(OW*FN) + k*(FN)+l;

					array_transpose[idx_transpose] = array[idx];
				}
			}
		}
	}
}

__global__ void Kernel_Transpose4d_forward(float* dev_transposeArray, float* dev_array,
	const int XN, const int OH, const int OW, const int FN) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*OH*OW*FN;
	int i, j, k, l;
	int idx_transposeArray, idx_array;

	while (tid < N)
	{
		idx4d(tid, XN, OH, OW, FN, i, j, k, l);
		idx_transposeArray = i*(FN*OH*OW) + l*(OH*OW) + j*(OW)+k;
		idx_array = i*(OH*OW*FN) + j*(OW*FN) + k*(FN)+l;

		dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

		tid += gridDim.x*blockDim.x;
	}

}

void Transpose4d_backward(float* array_transpose, float* array,
	const int XN, const int XC, const int OH, const int OW) {

	int i, j, k, l;
	int idx_transpose, idx;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < OH; k++) {
				for (l = 0; l < OW; l++) {

					idx_transpose = i*(OH*OW*XC) + k*(OW*XC) + l*(XC)+j;
					idx = i*(XC*OH*OW) + j*(OH*OW) + k*(OW)+l;

					array_transpose[idx_transpose] = array[idx];

				}
			}
		}
	}

}

__global__ void Kernel_Transpose4d_backward(float* dev_transposeArray, float* dev_array,
	const int XN, const int FN, const int OH, const int OW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*FN*OH*OW;
	int i, j, k, l;
	int idx_transposeArray, idx_array;

	while (tid < N)
	{
		idx4d(tid, XN, FN, OH, OW, i, j, k, l);
		idx_transposeArray = i*(OH*OW*FN) + k*(OW*FN) + l*(FN)+j;
		idx_array = i*(FN*OH*OW) + j*(OH*OW) + k*(OW)+l;

		dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

		tid += gridDim.x*blockDim.x;
	}

}

void Transpose4d(char txt, float* array_transpose, float* array,
	const int d1, const int d2, const int d3, const int d4) {

	int XN, OH, OW, FN, XC;

	switch (txt)
	{
	case 'f':
		XN = d1;
		OH = d2;
		OW = d3;
		FN = d4;
		Transpose4d_forward(array_transpose, array, XN, OH, OW, FN);

		break;
	case 'b':
		XN = d1;
		XC = d2;
		OH = d3;
		OW = d4;
		Transpose4d_backward(array_transpose, array, XN, XC, OH, OW);

		break;
	default:
		cout << "Error for 'txt' variable in Transpose4d(cpu)!" << endl;
		break;
	}

}

void Transpose4d_gpu(char txt, float* dev_transposeArray, float* dev_array,
	const int d1, const int d2, const int d3, const int d4) {

	int XN, OH, OW, FN;

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	switch (txt)
	{
	case 'f':
		XN = d1;
		OH = d2;
		OW = d3;
		FN = d4;
		if (XN == 1) {
			Transpose2d_gpu(dev_transposeArray, dev_array, OH*OW, FN);
		}
		else {
			Kernel_Transpose4d_forward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, OH, OW, FN);
			cudaDeviceSynchronize();
			gpuErrchk(cudaGetLastError());
		}
		//Kernel_Transpose4d_forward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, OH, OW, FN);
		//cudaDeviceSynchronize();
		//gpuErrchk(cudaGetLastError());
		break;
	case 'b':
		XN = d1;
		FN = d2;
		OH = d3;
		OW = d4;
		if (XN == 1) {
			Transpose2d_gpu(dev_transposeArray, dev_array, FN, OH*OW);
		}
		else {
			Kernel_Transpose4d_backward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, FN, OH, OW);
			cudaDeviceSynchronize();
			gpuErrchk(cudaGetLastError());
		}
		//Kernel_Transpose4d_backward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, FN, OH, OW);
		//cudaDeviceSynchronize();
		//gpuErrchk(cudaGetLastError());
		break;
	default:
		cout << "Error for 'txt' variable in Transpose4d(gpu)!" << endl;
		break;
	}

}

void Transpose6d_forward(float* array_transpose, float* array,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW) {

	int i, j, k, l, m, n;
	int idx_transpose;
	int idx;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < XC; j++) {
			for (k = 0; k < FH; k++) {
				for (l = 0; l < FW; l++) {
					for (m = 0; m < OH; m++) {
						for (n = 0; n < OW; n++) {

							idx_transpose = i*(OH*OW*XC*FH*FW) + m*(OW*XC*FH*FW) + n*(XC*FH*FW) + j*(FH*FW) + k*(FW)+l;
							idx = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;

							array_transpose[idx_transpose] = array[idx];
						}
					}
				}
			}
		}
	}


}

__global__ void Kernel_Transpose6d_forward(float* dev_transposeArray, float* dev_array,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW) {


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*FH*FW*OH*OW;
	int i, j, k, l, m, n;

	int idx_transposeArray;
	int idx_array;

	while (tid < N)
	{
		idx6d(tid, XN, XC, FH, FW, OH, OW, i, j, k, l, m, n);
		idx_transposeArray = i*(OH*OW*XC*FH*FW) + m*(OW*XC*FH*FW) + n*(XC*FH*FW) + j*(FH*FW) + k*(FW)+l;
		idx_array = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;

		dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

		tid += gridDim.x *blockDim.x;
	}

}

void Transpose6d_backward(float* array_transpose, float* array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {

	int i, j, k, l, m, n;
	int idx_transpose;
	int idx;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					for (m = 0; m < FH; m++) {
						for (n = 0; n < FW; n++) {

							idx_transpose = i*(XC*FH*FW*OH*OW) + l*(FH*FW*OH*OW) + m*(FW*OH*OW) + n*(OH*OW) + j*(OW)+k;
							idx = i*(OH*OW*XC*FH*FW) + j*(OW*XC*FH*FW) + k*(XC*FH*FW) + l*(FH*FW) + m*(FH)+n;

							array_transpose[idx_transpose] = array[idx];
						}
					}
				}
			}
		}
	}

}

__global__ void Kernel_Transpose6d_backward(float* dev_transposeArray, float* dev_array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*OH*OW*XC*FH*FW;
	int i, j, k, l, m, n;

	int idx_transposeArray;
	int idx_array;

	while (tid < N)
	{
		idx6d(tid, XN, OH, OW, XC, FH, FW, i, j, k, l, m, n);
		idx_transposeArray = i*(XC*FH*FW*OH*OW) + l*(FH*FW*OH*OW) + m*(FW*OH*OW) + n*(OH*OW) + j*(OW)+k;
		idx_array = i*(OH*OW*XC*FH*FW) + j*(OW*XC*FH*FW) + k*(XC*FH*FW) + l*(FH*FW) + m*(FH)+n;

		dev_transposeArray[idx_transposeArray] = dev_array[idx_array];

		tid += gridDim.x*blockDim.x;
	}

}

void Transpose6d(char txt, float* array_transpose, float* array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6) {

	int XN, OH, OW, FN, XC, FH, FW;
	int i, j, k, l;

	switch (txt)
	{
	case 'f':
		XN = d1;
		XC = d2;
		FH = d3;
		FW = d4;
		OH = d5;
		OW = d6;
		Transpose6d_forward(array_transpose, array, XN, XC, FH, FW, OH, OW);

		break;
	case 'b':
		XN = d1;
		OH = d2;
		OW = d3;
		XC = d4;
		FH = d5;
		FW = d6;
		Transpose6d_backward(array_transpose, array, XN, OH, OW, XC, FH, FW);

		break;
	default:
		cout << "Error for 'txt' variable in Transpose6d(cpu)!" << endl;
		break;
	}

}

void Transpose6d_gpu(char txt, float* dev_transposeArray, float* dev_array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6) {

	int XN, OH, OW, FN, XC, FH, FW;
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	switch (txt)
	{
	case 'f':
		XN = d1;
		XC = d2;
		FH = d3;
		FW = d4;
		OH = d5;
		OW = d6;
		if (XN == 1) {
			Transpose2d_gpu(dev_transposeArray, dev_array, XC*FH*FW, OH*OW);
		}
		else {
			Kernel_Transpose6d_forward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, XC, FH, FW, OH, OW);
			cudaDeviceSynchronize();
			gpuErrchk(cudaGetLastError());
		}
		//Kernel_Transpose6d_forward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, XC, FH, FW, OH, OW);
		//cudaDeviceSynchronize();
		//gpuErrchk(cudaGetLastError());

		break;
	case 'b':
		XN = d1;
		OH = d2;
		OW = d3;
		XC = d4;
		FH = d5;
		FW = d6;

		if (XN == 1) {
			Transpose2d_gpu(dev_transposeArray, dev_array, OH*OW, XC*FH*FW);
		}
		else {
			Kernel_Transpose6d_backward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, OH, OW, XC, FH, FW);
			cudaDeviceSynchronize();
			gpuErrchk(cudaGetLastError());
		}
		//Kernel_Transpose6d_backward << < dimGrid, dimBlock >> > (dev_transposeArray, dev_array, XN, OH, OW, XC, FH, FW);
		//cudaDeviceSynchronize();
		//gpuErrchk(cudaGetLastError());
		break;
	default:
		cout << "Error for 'txt' variable in Transpose6d(gpu)!" << endl;
		break;
	}
}





void Argmax(int* argMax, float** array, const int r, const int c) {


	int idx;
	float temp;

	for (int i = 0; i < r; i++) {
		idx = 0;
		temp = 0.0;

		for (int j = 0; j < c; j++) {

			if (array[i][j] > temp) {
				temp = array[i][j];
				idx = j;
			}

		}

		argMax[i] = idx;
	}

}

__global__ void Kernel_Argmax(int* dev_argMax, float* dev_array, const int r, const int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= r) return;

	int idx;
	float temp = 0.0;

	for (int j = 0; j < c; j++) {
		if (dev_array[i*c + j] > temp) {
			temp = dev_array[i*c + j];
			idx = j;
		}
	}
	dev_argMax[i] = idx;

}

void Argmax_gpu(int* dev_argMax, float* dev_array, const int r, const int c) {

	dim3 dimBlock(BLOCK_SIZE_X);
	dim3 dimGrid((r + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Argmax_gpu'!" << endl;
	}

	Kernel_Argmax << < dimGrid, dimBlock >> > (dev_argMax, dev_array, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

void Max(float* array_max, int* arg_max, float* array,
	const int r, const int c) {

	float temp;
	int idx;

	for (int i = 0; i < r; i++) {

		idx = 0;
		temp = 0.0;

		for (int j = 0; j < c; j++) {

			if (array[i*c + j] > temp) {
				temp = array[i*c + j];
				idx = j;
			}

		}

		arg_max[i] = idx;
		array_max[i] = temp;
	}

}

__global__ void Kernel_Max(float* dev_arrayMax, int* dev_argMax, float* dev_array,
	const int r, const int c) {


	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r;
	int idx, i;
	float temp;


	while (tid < N)
	{
		i = tid;
		temp = 0.;
		idx = 0;

		for (int j = 0; j < c; j++) {

			if (j == 0) temp = dev_array[i*c + j], idx = 0;
			else if (dev_array[i*c + j] > temp) temp = dev_array[i*c + j], idx = j;

		}

		dev_argMax[i] = idx;
		dev_arrayMax[i] = temp;


		tid += gridDim.x*blockDim.x;
	}


}

void Max_gpu(float* dev_arrayMax, int* dev_argMax, float* dev_array,
	const int r, const int c) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Max << < dimGrid, dimBlock >> > (dev_arrayMax, dev_argMax, dev_array, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Avg(float* array_avg, float* array,
	const int r, const int c)
{

	float sum;

	for (int i = 0; i < r; i++) {

		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += array[i*c + j];
		}

		array_avg[i] = sum / c;
	}

}

__global__ void Kernel_Avg(float* dev_arrayMax, float* dev_array,
	const int r, const int c) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r;
	float sum;
	int i;

	while (tid < N)
	{
		i = tid;
		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += dev_array[i*c + j];
		}

		dev_arrayMax[i] = sum / c;

		tid += gridDim.x*blockDim.x;
	}



}

void Avg_gpu(float* dev_arrayMax, float* dev_array,
	const int r, const int c) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Avg << < dimGrid, dimBlock >> > (dev_arrayMax, dev_array, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Function1_poolingBackward(float* dmax, int* arg_max, float* array,
	const int i_dmax, const int j_dmax) {

	int i, j;
	int r = i_dmax, c = j_dmax;

	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			dmax[i*c + j] = 0;
		}

		dmax[i*c + arg_max[i]] = array[i];
	}

}

__global__ void Kernel_Function1_poolingBackward(float* dev_dmax, int* dev_argMax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = i_dmax*j_dmax;
	int i, j;

	while (tid < N)
	{
		idx2d(tid, i_dmax, j_dmax, i, j);
		dev_dmax[i*j_dmax + j] = 0;
		dev_dmax[i*j_dmax + (dev_argMax[i])] = dev_flattenDout[i];

		tid += gridDim.x*blockDim.x;
	}

}

void Function1_poolingBackward_gpu(float* dev_dmax, int* dev_argMax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function1_poolingBackward << < dimGrid, dimBlock >> > (dev_dmax, dev_argMax, dev_flattenDout, i_dmax, j_dmax);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Function1_poolingBackward_avg(float* dmax, float* array,
	const int i_dmax, const int j_dmax)
{

	int i, j;
	int r = i_dmax, c = j_dmax;

	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			dmax[i*c + j] = array[i] / c;
		}
	}

}

__global__ void Kernel_Function1_poolingBackward_avg(float* dev_dmax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = i_dmax*j_dmax;
	int i, j;
	while (tid < N)
	{
		idx2d(tid, i_dmax, j_dmax, i, j);

		dev_dmax[i*j_dmax + j] = dev_flattenDout[i] / j_dmax;
		tid += gridDim.x*blockDim.x;
	}

}

void Function1_poolingBackward_avg_gpu(float* dev_dmax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function1_poolingBackward_avg << < dimGrid, dimBlock >> > (dev_dmax, dev_flattenDout, i_dmax, j_dmax);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Function2_poolingBackward(float** dcol, float** dmax,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW) {


	int i, j, k, l, m, n;

	for (i = 0; i < XN; i++) {
		for (j = 0; j < OH; j++) {
			for (k = 0; k < OW; k++) {
				for (l = 0; l < XC; l++) {
					for (m = 0; m < FH; m++) {
						for (n = 0; n < FW; n++) {
							dcol[i*(OH*OW) + j*(OW)+k][l*(FH*FW) + m*(FH)+n] = dmax[i*(OH*OW*XC) + j*(OW*XC) + k*(XC)+l][m*(FW)+n];
						}
					}
				}
			}
		}
	}

}

__global__ void Kernel_Function_reluForward(float* dev_x, int* dev_index, const int size) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	while (tid < N)
	{

		if (dev_x[tid] > 0) dev_index[tid] = 1;
		else dev_index[tid] = 0;

		dev_x[tid] *= dev_index[tid];

		tid += gridDim.x*blockDim.x;
	}


}

void Function_reluForward_gpu(float* dev_x, int* dev_index, const int size) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_reluForward << < dimGrid, dimBlock >> > (dev_x, dev_index, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function_reluBackward(float* dev_dout, int* dev_index, const int size) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	while (tid < N)
	{
		dev_dout[tid] *= dev_index[tid];
		tid += gridDim.x*blockDim.x;
	}

}

void Function_reluBackward_gpu(float* dev_dout, int* dev_index, const int size) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_reluBackward << < dimGrid, dimBlock >> > (dev_dout, dev_index, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_softmaxBackward(float* dev_dx, float* dev_y, int* dev_t,
	const int r, const int c) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r*c;

	while (tid < N)
	{
		dev_dx[tid] = (dev_y[tid] - dev_t[tid]) / r;
		tid += gridDim.x*blockDim.x;
	}
}

void Function_softmaxBackward_gpu(float* dev_dx, float* dev_y, int* dev_t,
	const int r, const int c) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_softmaxBackward << < dimGrid, dimBlock >> > (dev_dx, dev_y, dev_t, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}


/*batch*/

__global__ void Kernel_Function_batch1(float* dev_x, float* dev_x_batch,
	const int BN, const int XC, const int XH, const int XW,
	int randomNumber) {

	int i, j, k, l;

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = BN*XC*XH*XW;
	if (tid >= N) return;

	idx4d(tid, BN, XC, XH, XW, i, j, k, l);

	int idx_batch = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;
	int idx = (i + randomNumber)*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;

	dev_x_batch[idx_batch] = dev_x[idx];



}

__global__ void Kernel_Function_batch2(int* dev_t, int* dev_t_batch,
	const int BN, const int ON, int randomNumber) {

	int i, j;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = BN*ON;
	if (tid >= N) return;

	idx2d(tid, BN, ON, i, j);

	int idx_batch = i*ON + j;
	int idx = (i + randomNumber)*ON + j;

	dev_t_batch[idx_batch] = dev_t[idx];
}

void Function_batch_gpu(float* dev_x, int* dev_t, float* dev_x_batch, int* dev_t_batch,
	const int BN, const int XC, const int XH, const int XW,
	const int ON, int randomNumber) {





	int size = BN*XC*XH*XW;
	dim3 dimBlock1(BLOCK_SIZE);
	dim3 dimGrid1((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid1.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_batch_gpu'!" << endl;
	}
	Kernel_Function_batch1 << < dimGrid1, dimBlock1 >> > (dev_x, dev_x_batch, BN, XC, XH, XW, randomNumber);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	size = BN*ON;
	dim3 dimBlock2(BLOCK_SIZE);
	dim3 dimGrid2((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid2.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_batch_gpu'!" << endl;
	}
	Kernel_Function_batch2 << < dimGrid2, dimBlock2 >> > (dev_t, dev_t_batch, BN, ON, randomNumber);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}



/*dropout*/

__global__ void Kernel_Function_dropoutinit(unsigned int seed, curandState_t* states, const int size) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;


	while (tid < N)
	{


		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the cpu */
			tid, /* the sequence number should be different for each core (unless you want all
				 cores to get the same sequence of numbers for some reason - use thread id! */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[tid]);

		tid += gridDim.x*blockDim.x;
	}


}

void Function_dropoutinit_gpu(unsigned int seed, curandState_t* states, const int size) {

	dim3 dimBlock(512);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_dropoutinit << < dimGrid, dimBlock >> > (seed, states, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Function_dropoutForward(float* dev_x, int* dev_index, const int size,
	float dropoutRatio, int train_flg,
	curandState_t* states) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	float randomNumber;

	while (tid < N)
	{

		if (train_flg == 1) {

			randomNumber = curand_uniform(&states[tid]);
			if (randomNumber > dropoutRatio) dev_index[tid] = 1;
			else dev_index[tid] = 0;

			dev_x[tid] *= dev_index[tid];
		}
		else {
			dev_x[tid] *= (1.0/* - dropoutRatio*/);
		}


		tid += gridDim.x*blockDim.x;
	}

}

void Function_dropoutForward_gpu(float* dev_x, int* dev_index, const int size,
	float dropoutRatio, int train_flg,
	curandState_t* states) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function_dropoutForward << < dimGrid, dimBlock >> > (dev_x, dev_index, size, dropoutRatio, train_flg, states);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function_dropoutBackward(float* dev_dout, int* dev_index, const int size) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;

	while (tid < N)
	{
		dev_dout[tid] *= dev_index[tid];
		tid += gridDim.x*blockDim.x;
	}

}

void Function_dropoutBackward_gpu(float* dev_dout, int* dev_index, const int size) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_dropoutBackward << < dimGrid, dimBlock >> > (dev_dout, dev_index, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}



/*skip connection*/

__global__ void Kernel_Function_sc(float* dev_x, float* dev_x_skip, int size) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;

	while (tid < N)
	{

		dev_x[tid] += dev_x_skip[tid];
		tid += gridDim.x*blockDim.x;
	}


}

void Function_sc_gpu(float* dev_x, float* dev_x_skip, int size) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function_sc << < dimGrid, dimBlock >> > (dev_x, dev_x_skip, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}




/*BN*/

__global__ void Kernel_Function_bninit(float* dev_gamma, const int DN) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= DN) return;

	dev_gamma[tid] = 1;

}

void Function_bninit_gpu(float* dev_gamma, const int DN) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_bninit'!" << endl;
	}

	Kernel_Function_bninit << < dimGrid, dimBlock >> > (dev_gamma, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void Kernel_Function1_bnForward(float* dev_mu, float* dev_x,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_x[tid];
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_mu[blockIdx.x] = cache[0] / XN;
	__syncthreads();



}

void Function1_bnForward_gpu(float* dev_mu, float* dev_x,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function1_bnForward_gpu'" << endl;
	}

	Kernel_Function1_bnForward << < DN, BLOCK_SIZE >> > (dev_mu, dev_x, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function2_bnForward(float* dev_xc, float* dev_x, float* dev_mu,
	const int XN, const int DN) {

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	int idx = i*DN + j;

	dev_xc[idx] = dev_x[idx] - dev_mu[j];

}

void Function2_bnForward_gpu(float* dev_xc, float* dev_x, float* dev_mu,
	const int XN, const int DN) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function2_bnForward_gpu'!" << endl;
	}

	Kernel_Function2_bnForward << < dimGrid, dimBlock >> > (dev_xc, dev_x, dev_mu, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function3_bnForward(float* dev_std, float* dev_xc,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_xc[tid] * dev_xc[tid];
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_std[blockIdx.x] = sqrtf(cache[0] / XN + 1e-7);
	__syncthreads();



}

void Function3_bnForward_gpu(float* dev_std, float* dev_xc,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function3_bnForward_gpu'" << endl;
	}

	Kernel_Function3_bnForward << < DN, BLOCK_SIZE >> > (dev_std, dev_xc, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function4_bnForward(float* dev_xn, float* dev_xc, float* dev_std,
	const int XN, const int DN) {

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	int idx = i*DN + j;

	dev_xn[idx] = dev_xc[idx] / dev_std[j];
}

void Function4_bnForward_gpu(float* dev_xn, float* dev_xc, float* dev_std,
	const int XN, const int DN) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function4_bnForward_gpu'!" << endl;
	}

	Kernel_Function4_bnForward << < dimGrid, dimBlock >> > (dev_xn, dev_xc, dev_std, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function5_bnForward(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;


	dev_running_mean[tid] = momentum * dev_running_mean[tid] + (1 - momentum) * dev_mu[tid];
	dev_running_var[tid] = momentum * dev_running_var[tid] + (1 - momentum) * dev_std[tid] * dev_std[tid];

}

void Function5_bnForward_gpu(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function5_bnForward_gpu'!" << endl;
	}

	//Kernel_Function5_bnForward << < dimGrid, dimBlock >> > (dev_running_mean, dev_running_var, dev_mu, dev_std, momentum, DN);
	//cudaDeviceSynchronize();
	//gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function6_bnForward(float* dev_x, float* dev_running_mean, float* dev_running_var,
	const int XN, const int DN) {


	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	unsigned int idx = i*DN + j;

	dev_x[idx] = (dev_x[idx] - dev_running_mean[j]) / sqrtf(dev_running_var[j] + 1e-7);

}

void Function6_bnForward_gpu(float* dev_x, float* dev_running_mean, float* dev_running_var,
	const int XN, const int DN) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function6_bnForward_gpu'!" << endl;
	}

	Kernel_Function6_bnForward << < dimGrid, dimBlock >> > (dev_x, dev_running_mean, dev_running_var, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function7_bnForward(float* dev_x, float* dev_out, float* dev_gamma, float* dev_beta,
	const int XN, const int DN) {


	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	unsigned int idx = i*DN + j;

	dev_x[idx] = dev_gamma[j] * dev_out[idx] + dev_beta[j];


}

void Function7_bnForward_gpu(float* dev_x, float* dev_out, float* dev_gamma, float* dev_beta,
	const int XN, const int DN) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function7_bnForward_gpu'!" << endl;
	}

	Kernel_Function7_bnForward << < dimGrid, dimBlock >> > (dev_x, dev_out, dev_gamma, dev_beta, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}




__global__ void Function_bnForward(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN) {

	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i >= DN) return;

	dev_running_mean[i] = momentum * dev_running_mean[i] + (1 - momentum) * dev_mu[i];
	dev_running_var[i] = momentum * dev_running_var[i] + (1 - momentum) * dev_std[i] * dev_std[i];

}

void Function_bnForward_gpu(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((DN + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_bnForward_test_gpu'!" << endl;
	}

	Function_bnForward << < dimGrid, dimBlock >> > (dev_running_mean, dev_running_var, dev_mu, dev_std, momentum, DN);
	gpuErrchk(cudaGetLastError());




}




__global__ void Kernel_Function1_bnBackward(float* dev_dbeta, float* dev_dout,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_dout[tid];
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_dbeta[blockIdx.x] = cache[0];
	__syncthreads();






}

void Function1_bnBackward_gpu(float* dev_dbeta, float* dev_dout,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function1_bnBackward_gpu'" << endl;
	}

	Kernel_Function1_bnBackward << < DN, BLOCK_SIZE >> > (dev_dbeta, dev_dout, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function2_bnBackward(float* dev_dgamma, float* dev_xn, float* dev_dout,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_xn[tid] * dev_dout[tid];
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_dgamma[blockIdx.x] = cache[0];
	__syncthreads();


}

void Function2_bnBackward_gpu(float* dev_dgamma, float* dev_xn, float* dev_dout,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function2_bnBackward_gpu'" << endl;
	}

	Kernel_Function2_bnBackward << < DN, BLOCK_SIZE >> > (dev_dgamma, dev_xn, dev_dout, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function3_bnBackward(float* dev_dxn, float* dev_gamma, float* dev_dout, float* dev_dxc, float* dev_std,
	const int XN, const int DN) {


	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	unsigned int idx = i*DN + j;

	dev_dxn[idx] = dev_gamma[j] * dev_dout[idx];
	dev_dxc[idx] = dev_dxn[idx] / dev_std[j];

}

void Function3_bnBackward_gpu(float* dev_dxn, float* dev_gamma, float* dev_dout, float* dev_dxc, float* dev_std,
	const int XN, const int DN) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function3_bnBackward_gpu'!" << endl;
	}

	Kernel_Function3_bnBackward << < dimGrid, dimBlock >> > (dev_dxn, dev_gamma, dev_dout, dev_dxc, dev_std, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function4_bnBackward(float* dev_dstd, float* dev_dxn, float* dev_xc, float* dev_std,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_dxn[tid] * dev_xc[tid] / (dev_std[blockIdx.x] * dev_std[blockIdx.x]);
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_dstd[blockIdx.x] = -cache[0];
	__syncthreads();

}

void Function4_bnBackward_gpu(float* dev_dstd, float* dev_dxn, float* dev_xc, float* dev_std,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function4_bnBackward_gpu'" << endl;
	}

	Kernel_Function4_bnBackward << < DN, BLOCK_SIZE >> > (dev_dstd, dev_dxn, dev_xc, dev_std, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function5_bnBackward(float* dev_dxc, float* dev_xc, float* dev_dstd, float* dev_std,
	const int XN, const int DN, int batch_size) {


	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	unsigned int idx = i*DN + j;

	float dvar = 0.5 * dev_dstd[j] / dev_std[j];
	dev_dxc[idx] = dev_dxc[idx] + (2.0 / batch_size) * dev_xc[idx] * dvar;

}

void Function5_bnBackward_gpu(float* dev_dxc, float* dev_xc, float* dev_dstd, float* dev_std,
	const int XN, const int DN, int batch_size) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function5_bnBackward_gpu'!" << endl;
	}

	Kernel_Function5_bnBackward << < dimGrid, dimBlock >> > (dev_dxc, dev_xc, dev_dstd, dev_std, XN, DN, batch_size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}

__global__ void Kernel_Function6_bnBackward(float* dev_dmu, float* dev_dxc,
	const int XN, const int DN) {


	unsigned int cacheIdx = threadIdx.x;
	__shared__ float cache[BLOCK_SIZE];
	cache[cacheIdx] = 0;
	__syncthreads();


	unsigned int tid = blockIdx.x + threadIdx.x * DN;


	if (threadIdx.x < XN) {
		cache[cacheIdx] = dev_dxc[tid];
		__syncthreads();
	}

	int k = blockDim.x / 2;
	while (k != 0) {
		if (cacheIdx < k) cache[cacheIdx] += cache[cacheIdx + k];
		__syncthreads();
		k /= 2;
	}

	if (cacheIdx == 0) dev_dmu[blockIdx.x] = cache[0];
	__syncthreads();


}

void Function6_bnBackward_gpu(float* dev_dmu, float* dev_dxc,
	const int XN, const int DN) {

	if (XN > BLOCK_SIZE) {
		cout << "Batch size(XN) > " << BLOCK_SIZE << " in 'Function6_bnBackward_gpu'" << endl;
	}

	Kernel_Function6_bnBackward << < DN, BLOCK_SIZE >> > (dev_dmu, dev_dxc, XN, DN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function7_bnBackward(float* dev_dout, float* dev_dxc, float* dev_dmu,
	const int XN, const int DN, int batch_size) {


	unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i >= XN || j >= DN) return;

	unsigned int idx = i*DN + j;

	dev_dout[idx] = dev_dxc[idx] - (dev_dmu[j] / batch_size);

}

void Function7_bnBackward_gpu(float* dev_dout, float* dev_dxc, float* dev_dmu,
	const int XN, const int DN, int batch_size) {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((XN + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (DN + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	if (dimGrid.x > MAX_GRID_SIZE || dimGrid.y > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function7_bnBackward_gpu'!" << endl;
	}

	Kernel_Function7_bnBackward << < dimGrid, dimBlock >> > (dev_dout, dev_dxc, dev_dmu, XN, DN, batch_size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


}


/*LRN*/

__global__ void Kernel_Function_lrnForward1(float* dev_x, float* dev_X, float* dev_y4,
	float myBias, float myAlpha, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW) {

	int i, j, k, l, n;
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	if (tid >= N) return;

	idx4d(tid, XN, XC, XH, XW, i, j, k, l);
	int idx = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;

	dev_X[idx] = dev_x[idx];

	float sum = 0;
	int idx_n;
	for (n = j - myDepth_radius; n <= j + myDepth_radius; n++) {

		if (n < 0 || n >= XC) continue;

		idx_n = i*(XC*XH*XW) + n*(XH*XW) + k*(XW)+l;
		sum += powf(dev_x[idx_n], 2);

	}

	dev_y4[idx] = (myBias + myAlpha * sum);


}

__global__ void Kernel_Function_lrnForward2(float* dev_x, float* dev_y4,
	float myBeta,
	const int size) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	if (tid >= N) return;

	dev_x[tid] /= powf(dev_y4[tid], myBeta);

}

void Function_lrnForward_gpu(float* dev_x, float* dev_X, float* dev_y4,
	float myBias, float myAlpha, float myBeta, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW) {


	int size = XN*XC*XH*XW;

	dim3 dimBlock1(BLOCK_SIZE);
	dim3 dimGrid1((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid1.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_lrnForward_gpu'!" << endl;
	}
	Kernel_Function_lrnForward1 << < dimGrid1, dimBlock1 >> > (dev_x, dev_X, dev_y4, myBias, myAlpha, myDepth_radius, XN, XC, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	dim3 dimBlock2(BLOCK_SIZE);
	dim3 dimGrid2((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid2.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_lrnForward_gpu'!" << endl;
	}
	Kernel_Function_lrnForward2 << < dimGrid2, dimBlock2 >> > (dev_x, dev_y4, myBeta, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());



}


__global__ void Kernel_Function_lrnBackward1(float* dev_dout, float* dev_dout_new, float* dev_X, float* dev_y4,
	float myAlpha, float myBeta, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW) {

	int i, j, k, l, n;
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	if (tid >= N) return;

	idx4d(tid, XN, XC, XH, XW, i, j, k, l);
	int idx = i*(XC*XH*XW) + j*(XH*XW) + k*(XW)+l;


	float sum = 0;
	int idx_n;
	for (n = j - myDepth_radius; n <= j + myDepth_radius; n++) {

		if (n < 0 || n >= XC) continue;

		idx_n = i*(XC*XH*XW) + n*(XH*XW) + k*(XW)+l;
		sum += (dev_X[idx_n] * dev_dout[idx_n]) / powf(dev_y4[idx_n], myBeta + 1);

	}

	dev_dout_new[idx] = dev_dout[idx] / powf(dev_y4[idx], myBeta) - 2.0*myAlpha*myBeta * dev_X[idx] * sum;

}

__global__ void Kernel_Function_lrnBackward2(float* dev_dout, float* dev_dout_new,
	const int size) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	if (tid >= N) return;

	dev_dout[tid] = dev_dout_new[tid];
}

void Function_lrnBackward_gpu(float* dev_dout, float* dev_dout_new, float* dev_X, float* dev_y4,
	float myAlpha, float myBeta, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW) {



	int size = XN*XC*XH*XW;
	dim3 dimBlock1(BLOCK_SIZE);
	dim3 dimGrid1((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid1.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_lrnBackward_gpu'!" << endl;
	}
	Kernel_Function_lrnBackward1 << < dimGrid1, dimBlock1 >> > (dev_dout, dev_dout_new, dev_X, dev_y4, myAlpha, myBeta, myDepth_radius, XN, XC, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	dim3 dimBlock2(BLOCK_SIZE);
	dim3 dimGrid2((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid2.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Function_lrnBackward_gpu'!" << endl;
	}
	Kernel_Function_lrnBackward2 << < dimGrid2, dimBlock2 >> > (dev_dout, dev_dout_new, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());



}




/*accuracy*/

__global__ void Kernel_Function_acc(float* dev_predict, int* dev_label, int* dev_acc_binary,
	int N, int C_label, int C_output, int H, int W) {


	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int N_ = N*C_label*H*W;
	int i, j, k, l, j_, idx_label, idx_predict, idx_max;
	float tmp;

	while (tid < N_)
	{

		idx4d(tid, N, C_output, H, W, i, j, k, l);


		for (j_ = 0; j_ < C_output; j_++)
		{
			idx_predict = i*(C_output*H*W) + j_*(H*W) + k*(W)+l;

			if (j_ == 0) tmp = dev_predict[idx_predict], idx_max = 0;
			else if (dev_predict[idx_predict] > tmp)
			{
				tmp = dev_predict[idx_predict];
				idx_max = j_;
			}
		}

		idx4d(tid, N, C_label, H, W, i, j, k, l);
		idx_label = i*(C_label*H*W) + j*(H*W) + k*(W)+l;


		if (dev_label[idx_label] == idx_max) dev_acc_binary[idx_label] = 1;
		else dev_acc_binary[idx_label] = 0;

		tid += gridDim.x*blockDim.x;
	}




}

void Function_acc_gpu(float* dev_predict, int* dev_label, int* dev_acc_binary,
	int* image_shape, int the_number_of_class) {

	int N = image_shape[0];
	int C_label = 1, C_output = the_number_of_class;
	int H = image_shape[2];
	int W = image_shape[3];

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);


	Kernel_Function_acc << < dimGrid, dimBlock >> > (dev_predict, dev_label, dev_acc_binary, N, C_label, C_output, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function_acc_dice(float* dev_predict, int* dev_label, int* dev_predict_binary, int label,
	int N, int C_label, int C_output, int H, int W) {


	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int N_ = N*C_label*H*W;
	int i, j, k, l, j_, idx_label, idx_predict, idx_max;
	float tmp;

	while (tid < N_)
	{


		idx4d(tid, N, C_output, H, W, i, j, k, l);

		for (j_ = 0; j_ < C_output; j_++)
		{
			idx_predict = i*(C_output*H*W) + j_*(H*W) + k*(W)+l;

			if (j_ == 0) tmp = dev_predict[idx_predict], idx_max = 0;
			else if (dev_predict[idx_predict] > tmp)
			{
				tmp = dev_predict[idx_predict];
				idx_max = j_;
			}
		}


		idx4d(tid, N, C_label, H, W, i, j, k, l);
		idx_label = i*(C_label*H*W) + j*(H*W) + k*(W)+l;

		if (idx_max == label) dev_predict_binary[idx_label] = 1;
		else dev_predict_binary[idx_label] = 0;

		if (dev_label[idx_label] != 0 && dev_label[idx_label] != 255) dev_label[idx_label] = 1;
		else dev_label[idx_label] = 0;



		tid += gridDim.x*blockDim.x;
	}
}

void Function_acc_dice_gpu(float* dev_predict, int* dev_label, int* dev_predict_binary, int label,
	int* image_shape, int the_number_of_class) {

	int N = image_shape[0];
	int C_label = 1, C_output = the_number_of_class;
	int H = image_shape[2];
	int W = image_shape[3];

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);


	Kernel_Function_acc_dice << < dimGrid, dimBlock >> > (dev_predict, dev_label, dev_predict_binary, label, N, C_label, C_output, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function_acc_iou(float* dev_predict, int* dev_predict_index,
	int N, int C_label, int C_output, int H, int W) {


	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int N_ = N*C_label*H*W;
	int i, j, k, l, j_, idx_label, idx_predict, idx_max;
	float tmp;

	while (tid < N_)
	{


		idx4d(tid, N, C_output, H, W, i, j, k, l);

		for (j_ = 0; j_ < C_output; j_++)
		{
			idx_predict = i*(C_output*H*W) + j_*(H*W) + k*(W)+l;

			if (j_ == 0) tmp = dev_predict[idx_predict], idx_max = 0;
			else if (dev_predict[idx_predict] > tmp)
			{
				tmp = dev_predict[idx_predict];
				idx_max = j_;
			}
		}


		idx4d(tid, N, C_label, H, W, i, j, k, l);
		idx_label = i*(C_label*H*W) + j*(H*W) + k*(W)+l;

		dev_predict_index[idx_label] = idx_max;


		tid += gridDim.x*blockDim.x;
	}
}

void Function_acc_iou_gpu(float* dev_predict, int* dev_predict_index,
	int* image_shape, int the_number_of_class) {

	int N = image_shape[0];
	int C_label = 1, C_output = the_number_of_class;
	int H = image_shape[2];
	int W = image_shape[3];

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_acc_iou << < dimGrid, dimBlock >> > (dev_predict, dev_predict_index, N, C_label, C_output, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

int** Function_confusion_matrix(/*int** confusion_matrix, */int* predict, int* gt, int size, int the_number_of_class)
{
	int** confusion_matrix = new int*[the_number_of_class];
	for (int i = 0; i < the_number_of_class; i++) confusion_matrix[i] = new int[the_number_of_class];
	for (int i = 0; i < the_number_of_class; i++) memset(confusion_matrix[i], 0, the_number_of_class * sizeof(int));



	//row(i):ground-truth image, column(j):predicted image
	for (int i = 0; i < the_number_of_class; i++)
	{
		for (int j = 0; j < the_number_of_class; j++)
		{


			for (int pixel = 0; pixel < size; pixel++)
			{
				if (gt[pixel] != 255 && gt[pixel] != 0)
				{

					if (gt[pixel] == i + 1 && predict[pixel] == j + 1) confusion_matrix[i][j] += 1;

				}
			}



		}
	}

	return confusion_matrix;
}

void accuracy_top5(float* x, const int size)
{
	set<int> index_top5;
	float temp = 0;
	int index;

	for (int n = 0; n < 5; n++)
	{
		temp = 0;
		for (int i = 0; i < size; i++)
		{
			if (x[i] > temp && index_top5.find(i) == index_top5.end())
			{
				temp = x[i];
				index = i;
			}
		}

		index_top5.insert(index);
	}
	set<int>::iterator iter;
	for (iter = index_top5.begin(); iter != index_top5.end(); iter++)
	{
		cout << "index of top5 : " << *iter << ", score : " << x[*iter] * 100 << "(%)" << endl;
	}

}


/*concat*/

__global__ void Kernel_Function_concatForward(float* dev_out, float* dev_x1, float* dev_x2,
	int N, int C1, int C2, int H, int W)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int C = C1 + C2;
	int N_max = N*C*H*W;
	int i, j, k, l, idx, idx_x1, idx_x2;


	while (tid < N_max)
	{
		idx4d(tid, N, C, H, W, i, j, k, l);
		idx = i*(C*H*W) + j*(H*W) + k*(W)+l;
		idx_x1 = i*(C1*H*W) + j*(H*W) + k*(W)+l;
		idx_x2 = i*(C2*H*W) + (j - C1)*(H*W) + k*(W)+l;

		if (j < C1)
		{
			dev_out[idx] = dev_x1[idx_x1];
		}
		else
		{
			dev_out[idx] = dev_x2[idx_x2];
		}


		tid += gridDim.x*blockDim.x;
	}

}

void Function_concatForward_gpu(float* dev_out, float* dev_x1, float* dev_x2,
	int N, int C1, int C2, int H, int W)
{
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function_concatForward << < dimGrid, dimBlock >> > (dev_out, dev_x1, dev_x2, N, C1, C2, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

__global__ void Kernel_Function_concatBackward(float* dev_dout1, float* dev_dout2, float* dev_dout,
	int N, int C1, int C2, int H, int W)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int C = C1 + C2;
	int N_max = N*C*H*W;
	int i, j, k, l, idx, idx_dout1, idx_dout2;

	while (tid < N_max)
	{

		idx4d(tid, N, C, H, W, i, j, k, l);
		idx = i*(C*H*W) + j*(H*W) + k*(W)+l;
		idx_dout1 = i*(C1*H*W) + j*(H*W) + k*(W)+l;
		idx_dout2 = i*(C2*H*W) + (j - C1)*(H*W) + k*(W)+l;

		if (j < C1)
		{
			dev_dout1[idx_dout1] = dev_dout[idx];
		}
		else
		{
			dev_dout2[idx_dout2] = dev_dout[idx];
		}


		tid += gridDim.x*blockDim.x;
	}

}

void Function_concatBackward_gpu(float* dev_dout1, float* dev_dout2, float* dev_dout,
	int N, int C1, int C2, int H, int W)
{

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	Kernel_Function_concatBackward << < dimGrid, dimBlock >> > (dev_dout1, dev_dout2, dev_dout, N, C1, C2, H, W);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}




/*optimizer*/

__global__ void Kernel_Function_update_sgd(float lr, float* dev_parameter, float* dev_gradient, int size) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;

	while (tid < N)
	{
		dev_parameter[tid] -= lr * dev_gradient[tid];
		tid += gridDim.x*blockDim.x;
	}


}

void Function_update_sgd_gpu(float lr, float* dev_parameter, float* dev_gradient, int size)
{

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function_update_sgd << < dimGrid, dimBlock >> > (lr, dev_parameter, dev_gradient, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

void Function_update_sgd_cpu(float lr, float* parameter, float* gradient, int size)
{
	for (int i = 0; i < size; i++)
		parameter[i] -= lr * gradient[i];
}


__global__ void Kernel_Function_update_rmsprop(float lr, float dr, float* dev_parameter, float* dev_gradient, float* dev_h, int size) {


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;

	while (tid < N)
	{
		dev_h[tid] *= dr;
		dev_h[tid] += (1 - dr) *dev_gradient[tid] * dev_gradient[tid];
		dev_parameter[tid] -= lr * dev_gradient[tid] / (sqrt(dev_h[tid]) + 1e-7);

		tid += gridDim.x*blockDim.x;
	}
}

void Function_update_rmsprop_gpu(float lr, float dr, float* dev_parameter, float* dev_gradient, float* dev_h, int size) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	Kernel_Function_update_rmsprop << < dimGrid, dimBlock >> > (lr, dr, dev_parameter, dev_gradient, dev_h, size);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}






//////////////////////////////////////////////////////// src ver2 ////////////////////////////////////////////////////////

//new and delete
template <typename _type>
void new_cpu(_type* &src, int buffer) {
	src = new _type[buffer];
	memset(src, 0, buffer * sizeof(_type));
}

template <typename _type>
void delete_cpu(_type* &src) {
	delete[] src;
	src = NULL;
}

template <typename _type>
void new_gpu(_type* &src, int buffer) {

	gpuErrchk(cudaMalloc((void**)&src, buffer * sizeof(_type)));
	gpuErrchk(cudaMemset(src, 0, buffer * sizeof(_type)));

}

template <typename _type>
void delete_gpu(_type* &src) {
	gpuErrchk(cudaFree(src));
	src = NULL;
}

float* padding(float* x, int pad, int N, int C, int H, int W) {

	int idx, idx_pad;
	int H_pad = H + 2 * pad;
	int W_pad = W + 2 * pad;

	int buffer = N*C*H_pad*W_pad;
	float* x_pad = NULL;
	new_cpu<float>(x_pad, buffer);


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < C; j++) {
			for (int k = 0; k < H; k++) {
				for (int l = 0; l < W; l++) {

					idx = i*(C*H*W) + j*(H*W) + k*(W)+l;
					idx_pad = i*(C*H_pad*W_pad) + j*(H_pad*W_pad) + (k + pad)*(W_pad)+(l + pad);

					x_pad[idx_pad] = x[idx];

				}
			}
		}
	}

	delete_cpu<float>(x);


	return x_pad;
}

__global__ void kernel_padding_forward(float* x_pad, float* x, int pad,
	int N, int C, int H, int W,
	int H_pad, int W_pad) {

	int i, j, k, l;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int _N = N*C*H*W;

	int idx_pad, idx;

	while (tid < _N)
	{
		idx4d(tid, N, C, H, W, i, j, k, l);
		idx_pad = i*(C*H_pad*W_pad) + j*(H_pad*W_pad) + (k + pad)*(W_pad)+(l + pad);
		idx = i*(C*H*W) + j*(H*W) + k*(W)+l;

		x_pad[idx_pad] = x[idx];

		tid += gridDim.x*blockDim.x;
	}
}

float* padding_gpu(float* x, int pad, int N, int C, int H, int W) {

	int H_pad = H + 2 * pad;
	int W_pad = W + 2 * pad;

	int buffer = N*C*H_pad*W_pad;
	float* x_pad = NULL;
	new_gpu<float>(x_pad, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_padding_forward << < dimGrid, dimBlock >> > (x_pad, x, pad, N, C, H, W, H_pad, W_pad);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(x);

	return x_pad;
}

float* padding(float* dx, int pad, int N, int C, int H, int W, int stride)
{

	int dH = H + 2 * pad + stride - 1;
	int dW = W + 2 * pad + stride - 1;

	int buffer = N*C*H*W;
	float* dx_pad = NULL;
	new_cpu<float>(dx_pad, buffer);

	int idx_dx, idx_dx_pad;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < C; j++) {
			for (int k = 0; k < H; k++) {
				for (int l = 0; l < W; l++) {

					idx_dx_pad = i*(C*H*W) + j*(H*W) + k*(W)+l;
					idx_dx = i*(C*dH*dW) + j*(dH*dW) + (k + pad)*(dW)+(l + pad);

					dx_pad[idx_dx_pad] = dx[idx_dx];
				}
			}
		}
	}

	delete_cpu<float>(dx);

	return dx_pad;

}

__global__ void kernel_padding_backward(float* dx_pad, float* dx, int pad,
	int N, int C, int H, int W,
	int dH, int dW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int _N = N*C*H*W;
	int i = 0, j = 0, k = 0, l = 0, idx_dx_pad, idx_dx;
	while (tid < _N)
	{
		idx4d(tid, N, C, H, W, i, j, k, l);
		idx_dx_pad = i*(C*H*W) + j*(H*W) + k*(W)+l;
		idx_dx = i*(C*dH*dW) + j*(dH*dW) + (k + pad)*(dW)+(l + pad);

		dx_pad[idx_dx_pad] = dx[idx_dx];

		tid += gridDim.x*blockDim.x;
	}

}

float* padding_gpu(float* dx, int pad, int N, int C, int H, int W, int stride) {

	int dH = H + 2 * pad + stride - 1;
	int dW = W + 2 * pad + stride - 1;

	int buffer = N*C*H*W;
	float* dx_pad = NULL;
	new_gpu<float>(dx_pad, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_padding_backward << < dimGrid, dimBlock >> > (dx_pad, dx, pad, N, C, H, W, dH, dW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(dx);
	return dx_pad;
}

float* stride_forward(float* img, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int buffer = XN*XC*FH*FW*OH*OW;
	float* col = NULL;
	new_cpu<float>(col, buffer);


	int y_max, x_max;
	int idx_col, idx_img;
	for (int i = 0; i < XN; i++) {
		for (int j = 0; j < XC; j++) {

			for (int k = 0; k < FH; k++) {
				y_max = k + stride*OH;

				for (int l = 0; l < FW; l++) {
					x_max = l + stride*OW;

					for (int a = k, int m = 0; a < y_max; a = a + stride, m++) {
						for (int b = l, int n = 0; b < x_max; b = b + stride, n++) {

							idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
							idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;

							col[idx_col] = img[idx_img];

						}
					}
				}
			}
		}
	}

	delete_cpu<float>(img);

	return col;
}

__global__ void kernel_stride_forward(float* col, float* img, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*FH*FW*OH*OW;
	int i, j, k, l, m, n, a, b;

	int idx_col;
	int idx_img;

	while (tid < N)
	{
		idx6d(tid, XN, XC, FH, FW, OH, OW, i, j, k, l, m, n);

		a = k + m*stride;
		b = l + n*stride;

		idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
		idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;

		col[idx_col] = img[idx_img];

		tid += gridDim.x * blockDim.x;
	}

}

float* stride_forward_gpu(float* img, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int buffer = XN*XC*FH*FW*OH*OW;
	float* col = NULL;
	new_gpu<float>(col, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_stride_forward << < dimGrid, dimBlock >> > (col, img, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(img);

	return col;
}

float* stride_backward(float* col, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int buffer = XN*XC*XH*XW;
	float* img = NULL;
	new_cpu<float>(img, buffer);


	int y_max, x_max;
	int idx_img, idx_col;
	for (int i = 0; i < XN; i++) {
		for (int j = 0; j < XC; j++) {

			for (int k = 0; k < FH; k++) {
				y_max = k + stride*OH;

				for (int l = 0; l < FW; l++) {
					x_max = l + stride*OW;

					for (int a = k, int m = 0; a < y_max; a = a + stride, m++) {
						for (int b = l, int n = 0; b < x_max; b = b + stride, n++) {

							idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;
							idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;

							img[idx_img] += col[idx_col];

						}
					}
				}
			}
		}
	}

	delete_cpu<float>(col);

	return img;

}

__global__ void kernel_stride_backward(float* img, float* col, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = XN*XC*XH*XW;
	int i, j, a, b, idx_img, idx_col;
	int k, l, m, n, temp;

	while (tid < N)
	{
		idx4d(tid, XN, XC, XH, XW, i, j, a, b);
		idx_img = i*(XC*XH*XW) + j*(XH*XW) + a*(XW)+b;

		for (k = 0; k < FH && k <= a; k++)
		{
			m = (a - k) / stride;
			temp = k + stride*m;
			if (temp != a || m >= OH)
				continue;

			for (l = 0; l < FW && l <= b; l++)
			{
				n = (b - l) / stride;
				temp = l + stride*n;
				if (temp != b || n >= OW)
					continue;

				idx_col = i*(XC*FH*FW*OH*OW) + j*(FH*FW*OH*OW) + k*(FW*OH*OW) + l*(OH*OW) + m*(OW)+n;
				img[idx_img] += col[idx_col];

			}
		}



		tid += gridDim.x*blockDim.x;
	}

}

float* stride_backward_gpu(float* col, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW) {

	int buffer = XN*XC*XH*XW;
	float* img = NULL;
	new_gpu<float>(img, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_stride_backward << < dimGrid, dimBlock >> > (img, col, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(col);
	return img;
}


//dim=6
float* transpose(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3, int idx_new_dim4, int idx_new_dim5) {

	int old_dims[6] = { _dim0, _dim1, _dim2, _dim3, _dim4, _dim5 };
	int new_dims[6] = { 0 };
	new_dims[0] = old_dims[idx_new_dim0];
	new_dims[1] = old_dims[idx_new_dim1];
	new_dims[2] = old_dims[idx_new_dim2];
	new_dims[3] = old_dims[idx_new_dim3];
	new_dims[4] = old_dims[idx_new_dim4];
	new_dims[5] = old_dims[idx_new_dim5];


	int i = 0, j = 0, k = 0, l = 0, m = 0, n = 0;
	int* old_idx[6] = { &i, &j, &k, &l, &m, &n };
	int* i_new = old_idx[idx_new_dim0];
	int* j_new = old_idx[idx_new_dim1];
	int* k_new = old_idx[idx_new_dim2];
	int* l_new = old_idx[idx_new_dim3];
	int* m_new = old_idx[idx_new_dim4];
	int* n_new = old_idx[idx_new_dim5];


	int buffer = _dim0*_dim1*_dim2*_dim3*_dim4*_dim5;
	float* x_transpose = NULL;
	new_cpu<float>(x_transpose, buffer);


	int idx, idx_transpose;
	for (i = 0; i < _dim0; i++) {
		for (j = 0; j < _dim1; j++) {
			for (k = 0; k < _dim2; k++) {
				for (l = 0; l < _dim3; l++) {
					for (m = 0; m < _dim4; m++) {
						for (n = 0; n < _dim5; n++) {

							idx_transpose = (*i_new) * (new_dims[1] * new_dims[2] * new_dims[3] * new_dims[4] * new_dims[5])
								+ (*j_new) * (new_dims[2] * new_dims[3] * new_dims[4] * new_dims[5])
								+ (*k_new) * (new_dims[3] * new_dims[4] * new_dims[5])
								+ (*l_new) * (new_dims[4] * new_dims[5])
								+ (*m_new) * (new_dims[5])
								+ (*n_new);
							idx = i*(_dim1*_dim2*_dim3*_dim4*_dim5) + j*(_dim2*_dim3*_dim4*_dim5) + k*(_dim3*_dim4*_dim5) + l*(_dim4*_dim5) + m*(_dim5)+n;

							x_transpose[idx_transpose] = x[idx];
						}
					}
				}
			}
		}
	}


	delete_cpu<float>(x);
	return x_transpose;
}

__global__ void kernel_transpose_6(float* x_transpose, float* x,
	int _dim0, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3, int idx_new_dim4, int idx_new_dim5) {


	int old_dims[6] = { _dim0, _dim1, _dim2, _dim3, _dim4, _dim5 };
	int new_dims[6] = { 0 };
	new_dims[0] = old_dims[idx_new_dim0];
	new_dims[1] = old_dims[idx_new_dim1];
	new_dims[2] = old_dims[idx_new_dim2];
	new_dims[3] = old_dims[idx_new_dim3];
	new_dims[4] = old_dims[idx_new_dim4];
	new_dims[5] = old_dims[idx_new_dim5];


	int i = 0, j = 0, k = 0, l = 0, m = 0, n = 0;
	int* old_idx[6] = { &i, &j, &k, &l, &m, &n };
	int* i_new = old_idx[idx_new_dim0];
	int* j_new = old_idx[idx_new_dim1];
	int* k_new = old_idx[idx_new_dim2];
	int* l_new = old_idx[idx_new_dim3];
	int* m_new = old_idx[idx_new_dim4];
	int* n_new = old_idx[idx_new_dim5];


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = _dim0*_dim1*_dim2*_dim3*_dim4*_dim5;

	int idx_transpose;
	int idx;

	while (tid < N)
	{
		idx6d(tid, _dim0, _dim1, _dim2, _dim3, _dim4, _dim5, i, j, k, l, m, n);
		idx_transpose = (*i_new) * (new_dims[1] * new_dims[2] * new_dims[3] * new_dims[4] * new_dims[5])
			+ (*j_new) * (new_dims[2] * new_dims[3] * new_dims[4] * new_dims[5])
			+ (*k_new) * (new_dims[3] * new_dims[4] * new_dims[5])
			+ (*l_new) * (new_dims[4] * new_dims[5])
			+ (*m_new) * (new_dims[5])
			+ (*n_new);
		idx = i*(_dim1*_dim2*_dim3*_dim4*_dim5) + j*(_dim2*_dim3*_dim4*_dim5) + k*(_dim3*_dim4*_dim5) + l*(_dim4*_dim5) + m*(_dim5)+n;


		x_transpose[idx_transpose] = x[idx];

		tid += gridDim.x *blockDim.x;
	}

}

float* transpose_gpu(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3, int idx_new_dim4, int idx_new_dim5) {

	int buffer = _dim0*_dim1*_dim2*_dim3*_dim4*_dim5;
	float* x_transpose = NULL;
	new_gpu<float>(x_transpose, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_transpose_6 << < dimGrid, dimBlock >> > (x_transpose, x,
		_dim0, _dim1, _dim2, _dim3, _dim4, _dim5,
		idx_new_dim0, idx_new_dim1, idx_new_dim2, idx_new_dim3, idx_new_dim4, idx_new_dim5);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	delete_gpu<float>(x);
	return x_transpose;

}


//dim=4
float* transpose(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3) {

	int old_dims[4] = { _dim0, _dim1, _dim2, _dim3 };
	int new_dims[4] = { 0 };
	new_dims[0] = old_dims[idx_new_dim0];
	new_dims[1] = old_dims[idx_new_dim1];
	new_dims[2] = old_dims[idx_new_dim2];
	new_dims[3] = old_dims[idx_new_dim3];



	int i = 0, j = 0, k = 0, l = 0;
	int* old_idx[4] = { &i, &j, &k, &l };
	int* i_new = old_idx[idx_new_dim0];
	int* j_new = old_idx[idx_new_dim1];
	int* k_new = old_idx[idx_new_dim2];
	int* l_new = old_idx[idx_new_dim3];



	int buffer = _dim0*_dim1*_dim2*_dim3;
	float* x_transpose = NULL;
	new_cpu<float>(x_transpose, buffer);


	int idx, idx_transpose;
	for (i = 0; i < _dim0; i++) {
		for (j = 0; j < _dim1; j++) {
			for (k = 0; k < _dim2; k++) {
				for (l = 0; l < _dim3; l++) {

					idx_transpose = (*i_new) * (new_dims[1] * new_dims[2] * new_dims[3])
						+ (*j_new) * (new_dims[2] * new_dims[3])
						+ (*k_new) * (new_dims[3])
						+ (*l_new);

					idx = i*(_dim1*_dim2*_dim3) + j*(_dim2*_dim3) + k*(_dim3)+l;

					x_transpose[idx_transpose] = x[idx];

				}
			}
		}
	}


	delete_cpu<float>(x);
	return x_transpose;
}

__global__ void kernel_transpose_4(float* x_transpose, float* x,
	int _dim0, int _dim1, int _dim2, int _dim3,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3) {


	int old_dims[6] = { _dim0, _dim1, _dim2, _dim3 };
	int new_dims[6] = { 0 };
	new_dims[0] = old_dims[idx_new_dim0];
	new_dims[1] = old_dims[idx_new_dim1];
	new_dims[2] = old_dims[idx_new_dim2];
	new_dims[3] = old_dims[idx_new_dim3];



	int i = 0, j = 0, k = 0, l = 0;
	int* old_idx[6] = { &i, &j, &k, &l };
	int* i_new = old_idx[idx_new_dim0];
	int* j_new = old_idx[idx_new_dim1];
	int* k_new = old_idx[idx_new_dim2];
	int* l_new = old_idx[idx_new_dim3];



	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = _dim0*_dim1*_dim2*_dim3;

	int idx_transpose;
	int idx;

	while (tid < N)
	{
		idx4d(tid, _dim0, _dim1, _dim2, _dim3, i, j, k, l);
		idx_transpose = (*i_new) * (new_dims[1] * new_dims[2] * new_dims[3])
			+ (*j_new) * (new_dims[2] * new_dims[3])
			+ (*k_new) * (new_dims[3])
			+ (*l_new);

		idx = i*(_dim1*_dim2*_dim3) + j*(_dim2*_dim3) + k*(_dim3)+l;

		x_transpose[idx_transpose] = x[idx];

		tid += gridDim.x *blockDim.x;
	}

}

float* transpose_gpu(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3) {

	int buffer = _dim0*_dim1*_dim2*_dim3;
	float* x_transpose = NULL;
	new_gpu<float>(x_transpose, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_transpose_4 << < dimGrid, dimBlock >> > (x_transpose, x,
		_dim0, _dim1, _dim2, _dim3,
		idx_new_dim0, idx_new_dim1, idx_new_dim2, idx_new_dim3);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	delete_gpu<float>(x);
	return x_transpose;

}


//dim=2
float* transpose(float* x,
	int _dim0, int _dim1) {

	int buffer = _dim0*_dim1;
	float* x_transpose = NULL;
	new_cpu<float>(x_transpose, buffer);

	int idx, idx_transpose;
	for (int i = 0; i < _dim0; i++) {
		for (int j = 0; j < _dim1; j++) {
			idx = i*_dim1 + j;
			idx_transpose = j*_dim0 + i;
			x_transpose[idx_transpose] = x[idx];
		}
	}

	delete_cpu<float>(x);
	return x_transpose;
}

__global__ void kernel_transpose_2(float* x_transpose, float* x,
	int _dim0, int _dim1) {



	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= _dim0 || j >= _dim1) return;

	int idx, idx_transpose;

	idx = i*_dim1 + j;
	idx_transpose = j*_dim0 + i;

	x_transpose[idx_transpose] = x[idx];
}

float* transpose_gpu(float* x,
	int _dim0, int _dim1) {

	int buffer = _dim0*_dim1;
	float* x_transpose = NULL;
	new_gpu<float>(x_transpose, buffer);

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid((_dim0 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (_dim1 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	kernel_transpose_2 << < dimGrid, dimBlock >> > (x_transpose, x, _dim0, _dim1);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	delete_gpu<float>(x);
	return x_transpose;

}


float* dot(float* A, float* B,
	int r, int c, int n) {

	int buffer = r*c;
	float* out = NULL;
	new_cpu<float>(out, buffer);

	float temp;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			temp = 0.0;
			for (int k = 0; k < n; k++) {
				temp += A[i*n + k] * B[k*c + j];
			}
			out[i*c + j] = temp;
		}
	}

	return out;
}

__global__ void kernel_dot(float* out, float* A, float* B,
	int r, int c, int n) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int N = r*c;
	int i, j;
	float temp, A_val, B_val;

	while (tid < N)
	{
		temp = 0.0;
		A_val = 0.0;
		B_val = 0.0;

		idx2d(tid, r, c, i, j);

		for (int k = 0; k < n; k++) {
			A_val = A[i*n + k];
			B_val = B[k*c + j];
			temp += A_val*B_val;
		}
		out[i*c + j] = temp;

		tid += gridDim.x*blockDim.x;
	}

}

float* dot_gpu(float* A, float* B,
	int r, int c, int n) {

	int buffer = r*c;
	float* out = NULL;
	new_gpu<float>(out, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_dot << < dimGrid, dimBlock >> > (out, A, B, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	return out;
}

void _dot(float* out, float* A, float* B,
	int r, int c, int n) {

	float temp;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			temp = 0.0;
			for (int k = 0; k < n; k++) {
				temp += A[i*n + k] * B[k*c + j];
			}
			out[i*c + j] = temp;
		}
	}
}

void _dot_gpu(float* out, float* A, float* B,
	int r, int c, int n) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_dot << < dimGrid, dimBlock >> > (out, A, B, r, c, n);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}


void sum_forward(float* x, float* b,
	int r, int c) {

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			x[i*c + j] += b[j];
		}
	}
}

__global__ void kernel_sum_forward(float* x, float* b,
	int r, int c) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r*c;
	int i = 0, j = 0;

	while (tid < N)
	{
		idx2d(tid, r, c, i, j);

		x[i*c + j] += b[j];
		tid += gridDim.x*blockDim.x;
	}

}

void sum_forward_gpu(float* x, float* b,
	int r, int c) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_sum_forward << < dimGrid, dimBlock >> > (x, b, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

void sum_backward(float* db, float* dout,
	int r, int c) {

	memset(db, 0, c * sizeof(float));
	for (int j = 0; j < c; j++) {
		for (int i = 0; i < r; i++) {
			db[j] += dout[i*c + j];
		}
	}

}

__global__ void kernel_sum_backward(float* db, float* dout,
	int r, int c) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = c;
	while (tid < N)
	{
		for (int i = 0; i < r; i++) {
			db[tid] += dout[i*c + tid];
		}

		tid += gridDim.x*blockDim.x;
	}


}

void sum_backward_gpu(float* db, float* dout,
	int r, int c) {

	cudaMemset(db, 0, c * sizeof(float));
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_sum_backward << < dimGrid, dimBlock >> > (db, dout, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid)
{
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void kernel_sum_backward_opt1(float* sum, float* dout, int r, int c) {


	__shared__ float sdata[(BLOCK_SIZE_opt / 2)];

	unsigned int tid = threadIdx.x;
	unsigned int i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
	//if (i >= r) return;

	for (int j = 0; j < c; j++) {

		sdata[tid] = dout[i*c + j] + dout[(i + blockDim.x)*c + j];
		__syncthreads();

		if (blockDim.x >= 512) {
			if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
		}
		if (blockDim.x >= 256) {
			if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
		}
		if (blockDim.x >= 128) {
			if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
		}

		if (tid < 32) warpReduce<BLOCK_SIZE_opt / 2>(sdata, tid);

		if (tid == 0) sum[blockIdx.x*c + j] = sdata[0];
		__syncthreads();

	}
}

__global__ void Kernel_Sum_backward_opt2(float* db, float* sum, int r_sum, int c) {

	unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= c) return;

	float temp = 0;
	for (int i = 0; i < r_sum; i++) {
		temp += sum[i*c + j];
	}

	db[j] = temp;
}

void sum_backward_gpu(float* db, float* dout,
	int r, int c, bool use_sharedMemory)
{
	int buffer = (r + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt * c;
	float* sum = NULL;
	new_gpu<float>(sum, buffer);


	dim3 dimBlock1(BLOCK_SIZE_opt / 2);		//halve the number of threads
	dim3 dimGrid1((r + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt);
	kernel_sum_backward_opt1 << < dimGrid1, dimBlock1 >> > (sum, dout, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	int r_sum = buffer / c;
	dim3 dimBlock2(BLOCK_SIZE_opt);
	dim3 dimGrid2((c + BLOCK_SIZE_opt - 1) / BLOCK_SIZE_opt);
	Kernel_Sum_backward_opt2 << < dimGrid2, dimBlock2 >> > (db, sum, r_sum, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(sum);

}

float* max_poolingForward(int* argMax, float* col,
	int r, int c)
{
	int buffer = r;
	float* out = NULL;
	new_cpu<float>(out, buffer);
	new_cpu<int>(argMax, buffer);
	float temp;
	int idx;

	for (int i = 0; i < r; i++) {

		idx = 0;
		temp = col[i*c + 0];

		for (int j = 1; j < c; j++) {

			if (col[i*c + j] > temp) {
				temp = col[i*c + j];
				idx = j;
			}

		}

		argMax[i] = idx;
		out[i] = temp;
	}


	delete_cpu<float>(col);
	return out;
}

float* max_poolingForward(float* col,
	int r, int c)
{
	int buffer = r;
	float* out = NULL;
	new_cpu<float>(out, buffer);
	float temp;

	for (int i = 0; i < r; i++) {

		temp = col[i*c + 0];

		for (int j = 1; j < c; j++) {

			if (col[i*c + j] > temp) {
				temp = col[i*c + j];
			}

		}
		out[i] = temp;
	}

	delete_cpu<float>(col);
	return out;
}

float* avg_poolingForward(float* col,
	int r, int c)
{
	int buffer = r;
	float* out = NULL;
	new_cpu<float>(out, buffer);


	float sum;
	for (int i = 0; i < r; i++) {

		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += col[i*c + j];
		}

		out[i] = sum / c;
	}

	delete_cpu<float>(col);
	return out;
}

float* max_poolingBackward(int* argMax, float* dout,
	int r, int c) {

	int buffer = r*c;
	float* dcol = NULL;
	new_cpu<float>(dcol, buffer);

	for (int i = 0; i < r; i++) {
		dcol[i*c + argMax[i]] = dout[i];
	}

	delete_cpu<float>(dout);
	delete_cpu<int>(argMax);
	return dcol;
}

float* avg_poolingBackward(float* dout,
	int r, int c)
{
	int buffer = r*c;
	float* dcol = NULL;
	new_cpu<float>(dcol, buffer);


	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			dcol[i*c + j] = dout[i] / c;
		}
	}

	delete_cpu<float>(dout);
	return dcol;
}

__global__ void kernel_max_poolingForward_training(float* out, int* argMax, float* col,
	int r, int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r;
	int idx;
	float temp;

	while (i < N)
	{
		temp = col[i*c + 0];
		idx = 0;

		for (int j = 1; j < c; j++) {

			if (col[i*c + j] > temp) {
				temp = col[i*c + j];
				idx = j;
			}
		}

		argMax[i] = idx;
		out[i] = temp;

		i += gridDim.x*blockDim.x;
	}

}

float* max_poolingForward_gpu(int* argMax, float* col,
	int r, int c) {

	int buffer = r;
	float* out = NULL;
	new_gpu<float>(out, buffer);
	new_gpu<int>(argMax, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_max_poolingForward_training << < dimGrid, dimBlock >> > (out, argMax, col, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());


	delete_gpu<float>(col);
	return out;
}

__global__ void kernel_max_poolingForward_inference(float* out, float* col,
	int r, int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r;
	float temp;

	while (i < N)
	{
		temp = col[i*c + 0];

		for (int j = 1; j < c; j++) {

			if (col[i*c + j] > temp) {
				temp = col[i*c + j];
			}
		}
		out[i] = temp;

		i += gridDim.x*blockDim.x;
	}

}

float* max_poolingForward_gpu(float* col,
	int r, int c) {

	int buffer = r;
	float* out = NULL;
	new_gpu<float>(out, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_max_poolingForward_inference << < dimGrid, dimBlock >> > (out, col, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(col);
	return out;
}

__global__ void kernel_avg_poolingForward(float* out, float* col,
	int r, int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r;
	float sum;

	while (i < N)
	{
		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += col[i*c + j];
		}

		out[i] = sum / c;

		i += gridDim.x*blockDim.x;
	}


}

float* avg_poolingForward_gpu(float* col,
	int r, int c) {

	int buffer = r;
	float* out = NULL;
	new_gpu<float>(out, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_avg_poolingForward << < dimGrid, dimBlock >> > (out, col, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(col);
	return out;
}

__global__ void kernel_max_poolingBackward(float* dcol, int* argMax, float* dout,
	int r, int c) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = r*c;
	int i, j;

	while (tid < N)
	{
		idx2d(tid, r, c, i, j);
		dcol[i*c + j] = 0;
		dcol[i*c + (argMax[i])] = dout[i];

		tid += gridDim.x*blockDim.x;
	}

}

float* max_poolingBackward_gpu(int* argMax, float* dout,
	int r, int c) {

	int buffer = r*c;
	float* dcol = NULL;
	new_gpu<float>(dcol, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_max_poolingBackward << < dimGrid, dimBlock >> > (dcol, argMax, dout, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(dout);
	delete_gpu<int>(argMax);
	return dcol;
}

__global__ void kernel_avg_poolingBackward(float* dcol, float* dout,
	int r, int c) {

	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = r*c;
	int i, j;
	while (tid < N)
	{
		idx2d(tid, r, c, i, j);

		dcol[i*c + j] = dout[i] / c;
		tid += gridDim.x*blockDim.x;
	}

}

float* avg_poolingBackward_gpu(float* dout,
	int r, int c) {

	int buffer = r*c;
	float* dcol = NULL;
	new_gpu<float>(dcol, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_avg_poolingBackward << < dimGrid, dimBlock >> > (dcol, dout, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<float>(dout);
	return dcol;

}

__global__ void kernel_reluForward_training(float* x, int* index, int size, float negative_slope) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	while (tid < N)
	{

		if (x[tid] > 0) index[tid] = 1;
		else x[tid] *= negative_slope;

		tid += gridDim.x*blockDim.x;
	}


}

__global__ void kernel_reluForward_inference(float* x, int size, float negative_slope) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	while (tid < N)
	{
		if (x[tid] <= 0) x[tid] *= negative_slope;
		tid += gridDim.x*blockDim.x;
	}


}

void reluForward_gpu(float* x, int* index, int size, float negative_slope) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	int buffer = size;
	new_gpu<int>(index, buffer);

	kernel_reluForward_training << < dimGrid, dimBlock >> > (x, index, size, negative_slope);

	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

void reluForward_gpu(float* x, int size, float negative_slope) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_reluForward_inference << < dimGrid, dimBlock >> > (x, size, negative_slope);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

__global__ void kernel_reluBackward(float* dout, int* index, int size, float negative_slope) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int N = size;
	while (tid < N)
	{
		if (!index[tid]) dout[tid] *= negative_slope;
		tid += gridDim.x*blockDim.x;
	}

}

void reluBackward_gpu(float* dout, int* index, int size, float negative_slope) {

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);

	kernel_reluBackward << < dimGrid, dimBlock >> > (dout, index, size, negative_slope);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	delete_gpu<int>(index);
}

void softmax(float* x, 
	int r, int c) {

	float temp1, temp2;
	for (int i = 0; i < r; i++) {
		temp1 = 0.;
		temp2 = 0.;

		for (int j = 0; j < c; j++)
		{
			temp1 = max(x[i*c + j], temp1);
		}

		for (int j = 0; j < c; j++)
		{
			x[i*c + j] = expf(x[i*c + j] - temp1);
			temp2 += x[i*c + j];
		}

		for (int j = 0; j < c; j++) x[i*c + j] /= temp2;
	}
}

__global__ void kernel_softmax(float* x, int r, int c) {

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= r) return;

	float temp1 = 0., temp2 = 0.;
	for (int j = 0; j < c; j++) temp1 = max(x[i*c + j], temp1);

	for (int j = 0; j < c; j++) {
		x[i*c + j] = expf(x[i*c + j] - temp1);
		temp2 += x[i*c + j];
	}

	for (int j = 0; j < c; j++) x[i*c + j] /= temp2;
}

void softmax_gpu(float* x,
	int r, int c) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((r + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (dimGrid.x > MAX_GRID_SIZE) {
		cout << "dimension of Grid exceeds " << MAX_GRID_SIZE << " in 'Softmax_gpu'!" << endl;
	}

	kernel_softmax << < dimGrid, dimBlock >> > (x, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

}

float CEE(float* x, int* t, 
	int r, int c) {

	float temp = 0;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {

			if (t[i*c + j] == 1) {		//one-hot encoding
				temp += log(x[i*c + j] + 1e-7);
				continue;
			}
		}
	}

	temp /= -r;
	return temp;
}

__global__ void kernel_CEE(float* x, int* t, float* loss, 
	int r, int c) {

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int N = r;
	float temp;
	while (i < N)
	{
		for (int j = 0; j < c; j++) {

			if (t[i*c + j] == 1) {
				temp = logf(x[i*c + j] + 1e-7);
				atomicAdd(loss, temp);
				continue;
			}
		}


		i += gridDim.x*blockDim.x;
	}

}

float CEE_gpu(float* x, int* t, float* loss, 
	int r, int c) {


	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	cudaMemset(loss, 0, sizeof(float));
	kernel_CEE << < dimGrid, dimBlock >> > (x, t, loss, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	float _loss = 0;
	cudaMemcpy(&_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
	_loss /= -r;

	return _loss;

}

__global__ void kernel_softmaxBackward(float* dx, float* y, int* t,
	int r, int c) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int N = r*c;

	while (tid < N)
	{
		dx[tid] = (y[tid] - t[tid]) / r;
		tid += gridDim.x*blockDim.x;
	}
}

float* softmaxBackward_gpu(float* y, int* t,
	int r, int c) {

	int buffer = r*c;
	float* dx = NULL;
	new_gpu<float>(dx, buffer);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE);
	kernel_softmaxBackward << < dimGrid, dimBlock >> > (dx, y, t, r, c);
	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());

	return dx;
}