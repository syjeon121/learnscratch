#pragma once

class Network
{
public:
	float buffer_total;
	int dev;
	int the_number_of_conv;
	int the_number_of_fc;
	int train_flg;

	//shape
	int input_shape[4];


	//pointer


	//layer object



	//parameter and gradient
	float** parameter, **gradient;
	int* parameter_shape;


	//function
	Network();
	~Network();

	void init(int* input_shape, int the_number_of_class, int dev, int train_flg);
	void init_layer();
	void init_param();

	float* predict_cpu(float* x);
	float* predict_gpu(float* x);

	float loss_cpu(float* x, int* t);
	float loss_gpu(float* x, int* t);

	float gradient_cpu(float* x, int* t);
	float gradient_gpu(float* x, int* t);

	void save_parameter_cpu();
	void save_parameter_gpu();
};

