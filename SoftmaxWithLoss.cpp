#include "SoftmaxWithLoss.h"


SoftmaxWithLoss::SoftmaxWithLoss()
{
}

void SoftmaxWithLoss::init(string name, int dim, int* x_shape, int dev) {

	this->name = name;
	cout << "name=" << name << endl;

	if (dim == 2) {
		XN = x_shape[0], cout << "N="	<< XN << endl;				
		IN = x_shape[1], cout << "IN="	<< IN << endl;				
	}
	else if (dim == 4) {
		XN = x_shape[0], cout << "N="	<< XN << endl;				//The number of input data
		XC = x_shape[1], cout << "C="	<< XC << endl;				//The number of input channel
		XH = x_shape[2], cout << "H="	<< XH << endl;				//The size of input height
		XW = x_shape[3], cout << "W="	<< XW << endl;				//The size of input weight
		IN = XC*XH*XW;
	}
	this->dev = dev;


	if (dev == 2) {
		gpuErrchk(cudaMalloc((void**)&loss, sizeof(float)));
		gpuErrchk(cudaMemset(loss, 0, sizeof(float)));
	}
}

float SoftmaxWithLoss::forward_cpu(float* x, int* t) {

	softmax(x, XN, IN);
	float _loss = 0;
	_loss = CEE(x, t, XN, IN);

	y = x;
	this->t = t;

	return _loss;
}

float SoftmaxWithLoss::forward_gpu(float* x, int* t) {


	softmax_gpu(x, XN, IN);
	float _loss = 0;
	_loss = CEE_gpu(x, t, loss, XN, IN);		

	y = x;
	this->t = t;

	return _loss;

}

float* SoftmaxWithLoss::backward_cpu(float dout) {

	float*dx = NULL;
	new_cpu<float>(dx, XN*IN);

	//one-hot-label=true
	for (int i = 0; i < size_dx; i++)
		dx[i] = (y[i] - t[i]) / XN;

	delete_cpu<float>(y);
	return dx;
}

float* SoftmaxWithLoss::backward_gpu(float dout) {

	float* dx = NULL;
	dx = softmaxBackward_gpu(y, t, XN, IN);

	delete_gpu<float>(y);
	return dx;
}

SoftmaxWithLoss::~SoftmaxWithLoss()
{
	if (dev == 2)
	{
		cudaFree(loss);
		loss = NULL;
	}

}
