#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <assert.h>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <set>
#include <string>
using namespace std;

#define MAX_GRID_SIZE 65535
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define GRID_SIZE 10
#define BLOCK_SIZE_opt 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "gpuassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define max(x1, x2) x1 < x2 ? x2 : x1



void Check(int* output_shape, float* dev_out);



void Dot(float* C, float* A, float* B, const int r, const int c, const int n);

void Dot_gpu(float* dev_C, float* dev_A, float* dev_B,
	const int r, const int c, const int n);

void Dot_coalescing1_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n);

void Dot_coalescing2_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n);

void Dot_reduction_gpu(float* dev_c, float* dev_a, float* dev_b,
	const int r, const int c, const int n,
	float* reduction);

void Dot_atomic_gpu(float* dev_C, float* dev_A, float* dev_B,
	const int r, const int c, const int n);

void Sum(char txt, float* A, float* B, const int r, const int c);

void Sum_gpu(char txt, float* dev_A, float* dev_B, const int r, const int c);

void Sum_gpu(char txt, float* dev_A, float* dev_B, const int r, const int c,
	float* dev_sum);

void Sum_gpu1(char txt, float* dev_A, float* dev_B, const int r, const int c,
	float* dev_partial, int size_partial);

float MSE(float** x1, float** x2, const int r, const int c);

void Softmax(float* x, const int r, const int c);

void Softmax_seg(float* x, const int size_category, const int size_spatial_feature_map);

void Softmax_gpu(float* dev_x, const int r, const int c);

void Softmax_seg_gpu(float* dev_x, const int size_category, const int size_spatial_feature_map);

void Softmax4d_gpu(float* dev_x, int N, int C, int H, int W);

void Softmax_gpu_shared(float* dev_x, const int XN, const int DN, float* dev_partialX4d, int size_partialX4d);



float CEE_seg(float* x, int* t, const int size_category, const int size_spatial_feature_map);

float CEE_gpu(float* dev_x, int* dev_t, float* dev_loss, const int r, const int c);

float CEE_seg_gpu(float* dev_x, int* dev_t, float* dev_loss,
	const int size_category, const int size_spatial_feature_map);

void Padding_forward(char txt, float* x_pad, float* x, const int pad,
	const int XN, const int XC, const int XH, const int XW);

void Padding_backward(char txt, float* dx_pad, float* dx, const int pad,
	const int XN, const int XC, const int XH, const int XW,
	const int dXH, const int dXW);

void Padding_forward_gpu(float* dev_x_pad, float*dev_X, const int pad,
	const int XN, const int XC, const int XH, const int XW);

void Padding_backward_gpu(float* dev_dx_pad, float* dev_dx, const int pad,
	const int XN, const int XC, const int XH, const int XW,
	const int dXH, const int dXW);

void Padding_transpose_forward(float* x_pad, float* x, int stride, int pad,
	int XN, int XC, int XH, int XW, int XH_pad, int XW_pad);

void Padding_transpose_backward(float* dx_pad, float* dx, int stride, int pad,
	int XN, int XC, int XH, int XW, int dXH, int dXW);

void Padding_transpose_forward_gpu(float* dev_x_pad, float* dev_x, int stride, int pad,
	int XN, int XC, int XH, int XW, int XH_pad, int XW_pad);

void Padding_transpose_backward_gpu(float* dev_dx_pad, float* dev_dx, int stride, int pad,
	int XN, int XC, int XH, int XW, int dXH, int dXW);

void Stride_forward(float* col, float* img, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW);

void Stride_forward_gpu(float* dev_col, float* dev_img, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW);

void Stride_backward(float* img, float* col, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW);

void Stride_backward_gpu(float* dev_img, float* dev_col, int stride,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW,
	const int XH, const int XW);

void Flatten6d(float* flattenX, float****** X, const int d1, const int d2, const int d3, const int d4, const int d5, const int d6);

void Flatten4d(float* flattenX, float**** X, const int d1, const int d2, const int d3, const int d4);

void Flatten2d(float* flattenX, float** X,
	const int d1, const int d2);

void Flatten2d_int(int* flattenX, int** X,
	const int d1, const int d2);

void Reshape6to2(float** reshapeArray, float****** array, const int XN, const int OH, const int OW, const int XC, const int FH, const int FW);

void Reshape6to2_gpu(float* dev_reshapeArray, float* dev_array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW,
	float* host_reshapeArray, int size_reshapeArray);

void Reshape6to2_poolingForward(float** reshapeArray, float****** array, const int XN, const int OH, const int OW, const int XC, const int FH, const int FW);

void Reshape4to2_forward(float** reshapeArray, float**** array,
	const int FN, const int FC, const int FH, const int FW);

void Reshape4to2_backward(float** reshapeArray, float**** array,
	const int XN, const int OH, const int OW, const int FN);

void Reshape4to2(char txt, float** reshapeArray, float**** array,
	const int d1, const int d2, const int d3, const int d4);

template <class T>
void Reshape4to2_forward(T** reshapeArray, T**** array,
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

template <class T>
void Reshape4to2_backward(T** reshapeArray, T**** array,
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

template <class T>
void Reshape4to2(char txt, T** reshapeArray, T**** array,
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
	const int XN, const int OH, const int OW, const int FN);

void Reshape2to4_backward(float**** reshapeArray, float** array,
	const int XN, const int XC, const int XH, const int XW);

void Reshape2to4(char txt, float**** reshapeArray, float** array,
	const int d1, const int d2, const int d3, const int d4);

template <class T>
void Reshape2to4_forward(T**** reshapeArray, T** array,
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

template <class T>
void Reshape2to4_backward(T**** reshapeArray, T** array,
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

template <class T>
void Reshape2to4(char txt, T**** reshapeArray, T** array,
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
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW);

void Reshape1to6(float****** reshapeArray, float* array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6);

void Reshape1to4(float**** reshapeArray, float* array, const int XN, const int OH, const int OW, const int XC);

void Reshape1to2(float** reshapeArray, float* array,
	const int d1, const int d2);

void Transpose2d(float* array_transpose, float* array, const int r, const int c);

void Transpose2d_gpu(float* dev_transposeArray, float* dev_array, const int r, const int c);

void Transpose4d_forward(float* array_transpose, float* array,
	const int XN, const int OH, const int OW, const int FN);

void Transpose4d_backward(float* array_transpose, float* array,
	const int XN, const int XC, const int OH, const int OW);

void Transpose4d(char txt, float* array_transpose, float* array,
	const int d1, const int d2, const int d3, const int d4);

void Transpose4d_gpu(char txt, float* dev_transposeArray, float* dev_array,
	const int d1, const int d2, const int d3, const int d4);

void Transpose6d_forward(float* array_transpose, float* array,
	const int XN, const int XC, const int FH, const int FW, const int OH, const int OW);

void Transpose6d_backward(float* array_transpose, float* array,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW);

void Transpose6d(char txt, float* array_transpose, float* array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6);

void Transpose6d_gpu(char txt, float* dev_transposeArray, float* dev_array,
	const int d1, const int d2, const int d3, const int d4, const int d5, const int d6);

void Argmax(int* argMax, float** array, const int r, const int c);

void Argmax_gpu(int* dev_argMax, float* dev_array, const int r, const int c);

void Max(float* array_max, int* arg_max, float* array,
	const int r, const int c);

void Max_gpu(float* dev_arrayMax, int* dev_argMax, float* dev_array,
	const int r, const int c);

void Avg(float* array_avg, float* array,
	const int r, const int c);

void Avg_gpu(float* dev_arrayMax, float* dev_array,
	const int r, const int c);

void Function1_poolingBackward(float* dmax, int* arg_max, float* array,
	const int i_dmax, const int j_dmax);

void Function1_poolingBackward_gpu(float* dev_dmax, int* dev_argMax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax);

void Function1_poolingBackward_avg(float* dmax, float* array,
	const int i_dmax, const int j_dmax);

void Function1_poolingBackward_avg_gpu(float* dev_dmax, float* dev_flattenDout,
	const int i_dmax, const int j_dmax);

void Function2_poolingBackward(float** dcol, float** dmax,
	const int XN, const int OH, const int OW, const int XC, const int FH, const int FW);

void Function_reluForward_gpu(float* dev_x, int* dev_index, const int size);

void Function_reluBackward_gpu(float* dev_dout, int* dev_index, const int size);

void Function_softmaxBackward_gpu(float* dev_dx, float* dev_y, int* dev_t,
	const int r, const int c);


/*Batch*/

void Function_batch_gpu(float* dev_x, int* dev_t, float* dev_x_batch, int* dev_t_batch,
	const int BN, const int XC, const int XH, const int XW,
	const int ON, int randomNumber);




/*Dropout*/


void Function_dropoutinit_gpu(unsigned int seed, curandState_t* states, const int size);

void Function_dropoutForward_gpu(float* dev_x, int* dev_index, const int size,
	float dropoutRatio, int train_flg,
	curandState_t* states);

void Function_dropoutBackward_gpu(float* dev_dout, int* dev_index, const int size);

/*Skip connection*/

void Function_sc_gpu(float* dev_x, float* dev_x_skip, int size);



/*BN*/

void Function_bninit_gpu(float* dev_gamma, const int DN);

void Function1_bnForward_gpu(float* dev_mu, float* dev_x,
	const int XN, const int DN);

void Function2_bnForward_gpu(float* dev_xc, float* dev_x, float* dev_mu,
	const int XN, const int DN);

void Function3_bnForward_gpu(float* dev_std, float* dev_xc,
	const int XN, const int DN);

void Function4_bnForward_gpu(float* dev_xn, float* dev_xc, float* dev_std,
	const int XN, const int DN);

void Function5_bnForward_gpu(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN);

void Function6_bnForward_gpu(float* dev_x, float* dev_running_mean, float* dev_running_var,
	const int XN, const int DN);

void Function7_bnForward_gpu(float* dev_x, float* dev_out, float* dev_gamma, float* dev_beta,
	const int XN, const int DN);




void Function_bnForward_gpu(float* dev_running_mean, float* dev_running_var, float* dev_mu, float* dev_std,
	float momentum, const int DN);




void Function1_bnBackward_gpu(float* dev_dbeta, float* dev_dout,
	const int XN, const int DN);

void Function2_bnBackward_gpu(float* dev_dgamma, float* dev_xn, float* dev_dout,
	const int XN, const int DN);

void Function3_bnBackward_gpu(float* dev_dxn, float* dev_gamma, float* dev_dout, float* dev_dxc, float* dev_std,
	const int XN, const int DN);

void Function4_bnBackward_gpu(float* dev_dstd, float* dev_dxn, float* dev_xc, float* dev_std,
	const int XN, const int DN);

void Function5_bnBackward_gpu(float* dev_dxc, float* dev_xc, float* dev_dstd, float* dev_std,
	const int XN, const int DN, int batch_size);

void Function6_bnBackward_gpu(float* dev_dmu, float* dev_dxc,
	const int XN, const int DN);

void Function7_bnBackward_gpu(float* dev_dout, float* dev_dxc, float* dev_dmu,
	const int XN, const int DN, int batch_size);



/*LRN*/

void Function_lrnForward_gpu(float* dev_x, float* dev_X, float* dev_y4,
	float myBias, float myAlpha, float myBeta, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW);

void Function_lrnBackward_gpu(float* dev_dout, float* dev_dout_new, float* dev_X, float* dev_y4,
	float myAlpha, float myBeta, int myDepth_radius,
	const int XN, const int XC, const int XH, const int XW);


/*Accuracy*/

void Function_acc_gpu(float* dev_predict, int* dev_label, int* dev_acc_binary,
	int* image_shape, int the_number_of_class);

void Function_acc_dice_gpu(float* dev_predict, int* dev_label, int* dev_predict_binary, int label,
	int* image_shape, int the_number_of_class);

void Function_acc_iou_gpu(float* dev_predict, int* dev_predict_index,
	int* image_shape, int the_number_of_class);

int** Function_confusion_matrix(int* predict, int* gt, int size, int the_number_of_class);

void accuracy_top5(float* x, const int size);

/*Concat*/
void Function_concatForward_gpu(float* dev_out, float* dev_x1, float* dev_x2,
	int N, int C1, int C2, int H, int W);

void Function_concatBackward_gpu(float* dev_dout1, float* dev_dout2, float* dev_dout,
	int N, int C1, int C2, int H, int W);


/*Optimizer*/

void Function_update_sgd_gpu(float lr, float* dev_parameter, float* dev_gradient, int size);

void Function_update_sgd_cpu(float lr, float* parameter, float* gradient, int size);

void Function_update_rmsprop_gpu(float lr, float dr, float* dev_parameter, float* dev_gradient, float* dev_h, int size);


//////////////////////////////////////////////////////// src ver2 ////////////////////////////////////////////////////////
template <typename _type>
void new_cpu(_type* &src, int buffer);

template <typename _type>
void delete_cpu(_type* &src);

template <typename _type>
void new_gpu(_type* &src, int buffer);

template <typename _type>
void delete_gpu(_type* &src);

float* padding(float* x, int pad, int N, int C, int H, int W);

float* padding_gpu(float* x, int pad, int N, int C, int H, int W);

float* padding(float* dx, int pad, int N, int C, int H, int W, int stride);

float* padding_gpu(float* dx, int pad, int N, int C, int H, int W, int stride);

float* stride_forward(float* img, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW);

float* stride_forward_gpu(float* img, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW);

float* stride_backward(float* col, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW);

float* stride_backward_gpu(float* col, int stride,
	int XN, int XC, int FH, int FW, int OH, int OW, int XH, int XW);

//dim=6
float* transpose(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3, int idx_new_dim4, int idx_new_dim5);

float* transpose_gpu(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3, int idx_new_dim4, int idx_new_dim5);

//dim=4
float* transpose(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3);

float* transpose_gpu(float* x,
	int _dim0, int _dim1, int _dim2, int _dim3,
	int idx_new_dim0, int idx_new_dim1, int idx_new_dim2, int idx_new_dim3);

//dim=2
float* transpose(float* x,
	int _dim0, int _dim1);

float* transpose_gpu(float* x,
	int _dim0, int _dim1);

float* dot(float* A, float* B,
	int r, int c, int n);

float* dot_gpu(float* A, float* B,
	int r, int c, int n);

void _dot(float* out, float* A, float* B,
	int r, int c, int n);

void _dot_gpu(float* out, float* A, float* B,
	int r, int c, int n);

void sum_forward(float* x, float* b,
	int r, int c);

void sum_forward_gpu(float* x, float* b,
	int r, int c);

void sum_backward(float* db, float* dout,
	int r, int c);

void sum_backward_gpu(float* db, float* dout,
	int r, int c);

void sum_backward_gpu(float* db, float* dout,
	int r, int c, bool use_sharedMemory);

float* max_poolingForward(int* argMax, float* col,
	int r, int c);

float* max_poolingForward(float* col,
	int r, int c);

float* avg_poolingForward(float* col,
	int r, int c);

float* max_poolingBackward(int* argMax, float* dout,
	int r, int c);

float* avg_poolingBackward(float* dout,
	int r, int c);

float* max_poolingForward_gpu(int* argMax, float* col,
	int r, int c);

float* max_poolingForward_gpu(float* col,
	int r, int c);

float* avg_poolingForward_gpu(float* col,
	int r, int c);

float* max_poolingBackward_gpu(int* argMax, float* dout,
	int r, int c);

float* avg_poolingBackward_gpu(float* dout,
	int r, int c);

void reluForward_gpu(float* x, int* index, int size, float negative_slope);

void reluForward_gpu(float* x, int size, float negative_slope);

void reluBackward_gpu(float* dout, int* index, int size, float negative_slope);

void softmax(float* x,
	int r, int c);

void softmax_gpu(float* x,
	int r, int c);

float CEE(float* x, int* t, 
	int r, int c);

float CEE_gpu(float* x, int* t, float* loss,
	int r, int c);

float* softmaxBackward_gpu(float* y, int* t,
	int r, int c);
