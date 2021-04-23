#include "Relu.h"

Relu::Relu()
{
}

void Relu::init(string name, int dim, int* input_shape, int dev, 
	int train_flg, float negative_slope) {

	this->name = name;
	cout << endl << "name=" << name << endl;

	size = 1;
	for (int i = 0; i < dim; i++) size *= input_shape[i];

	this->train_flg = train_flg;
	this->negative_slope = negative_slope;
}

void Relu::forward_cpu(float* x) {

	if (train_flg == 1) {
		new_cpu<int>(index, size);

		for (int i = 0; i < size; i++)
		{
			if (x[i] > 0) index[i] = 1;
			else x[i] *= negative_slope;
		}
	}
	else {
		for (int i = 0; i < size; i++)
		{
			if (x[i] <= 0) x[i] *= negative_slope;
		}
	}
}

void Relu::forward_gpu(float* x) {

	if (train_flg == 1) reluForward_gpu(x, index, size, negative_slope);
	else reluForward_gpu(x, size, negative_slope);
}

void Relu::backward_cpu(float* dout) {

	for (int i = 0; i < size; i++)
	{
		if (!index[i]) dout[i] *= negative_slope;
	}

	delete_cpu<int>(index);
}

void Relu::backward_gpu(float* dout) {

	reluBackward_gpu(dout, index, size, negative_slope);

}

Relu::~Relu() {


}
