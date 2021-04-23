#pragma once
#include "Function.cuh"

class Fully_connected
{
public:

	string name;
	int nodeNum;
	int dev;
	int train_flg;
	float buffer;


	int XN, XC, XH, XW;
	int IN, WN;
	int output_size;
	int input_size;

	int size_w, size_b;
	int size_dw, size_db;



	float *w, *b;
	float *dw, *db;
	float *x;
	float* host_w, *host_b;




	void init(string name, int dim, int* x_shape, int num_output, int dev,
		int train_flg, int previous_size=0);
	void init_size();
	void init_cpu();
	void init_gpu();
	void init_trained_parameter(ifstream& fin_w, ifstream& fin_b);
	void save_trained_parameter(ofstream& fout_w, ofstream& fout_b);
	float* forward_cpu(float* x);
	float* forward_gpu(float* x);
	float* backward_cpu(float* dout);
	float* backward_gpu(float* dout);


	Fully_connected();
	~Fully_connected();










	//==================



	//char txt;



	///*2D*/
	//

	//int size_W, size_b, size_out, size_dW, size_db, size_transposeX, size_transposeW, size_dX;

	//float** W, **out, **dX, **b, **W_T, **dW, **db, **X, **X_T, **dout_T;

	//float* host_W, *host_b;
	//float* dev_W, *dev_out, *dev_b, *dev_dW, *dev_db, *dev_transposeX, *dev_X, *dev_transposeW, *dev_dX;


	//void Init2d(int* x_shape, int hiddenSize, int* w_shape, int dev, int* out_shape);
	//void Init2d_CPU();
	//void Init2d_GPU();
	//float** forward2d_CPU(float** X);
	//float* forward2d_GPU(float* dev_x);
	//float** backward2d_CPU(float** dout);
	//float* backward2d_GPU(float* dev_dout);




	///*4D*/
	//int myDev;


	//
	//int i_reshapeX4d, j_reshapeX4d, i_W4d, j_W4d, j_b4d, i_out4d, j_out4d;
	//int i_transposeW4d, j_transposeW4d;
	//int i_transposeDout4d, j_transposeDout4d;
	//int i_transposeX4d, j_transposeX4d;
	//int i_dW4d, j_dW4d;
	//int j_db4d;
	//int i_dX4d, j_dX4d;


	//float** reshapeX4d, **W4d, **transposeW4d, **b4d, **out4d, **transposeDout4d, **transposeX4d;
	//float** dW4d, **db4d, **dX4d, ****reshapedX4d;



	//int size_W4d, size_out4d;
	//int size_b4d;
	//int size_X4d, size_transposeX4d;
	//int size_dW4d, size_db4d, size_transposeW4d;
	//int size_dX4d;

	//float* dev_W4d, *host_W4d, *dev_out4d;
	//float* dev_b4d, *host_b4d;
	//float* dev_X4d, *dev_transposeX4d;
	//float* dev_dW4d, *dev_db4d, *dev_transposeW4d;
	//float* dev_dX4d;







	//void Init4d(int* x_shape, int hiddenSize, int *w_shape, int dev, int* out_shape);
	//void Init4d_index();
	//void Init4d_CPU();
	//void Init4d_GPU();

	//float** forward4to2_CPU(float**** x);
	//float* forward4to2_GPU(float* dev_x);
	//float**** backward2to4_CPU(float** dout);
	//float* backward2to4_GPU(float* dev_dout);
};

