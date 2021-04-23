
#include "Pooling.h"

Pooling::Pooling()
{

}

void Pooling::init(string name, int* x_shape, int kernel_size, const int stride, int dev, 
	string type, int train_flg) {

	this->name = name;
	cout << "name=" << name << endl;

	XN = x_shape[0], cout << "N=" << XN << endl;							//The number of input data
	XC = x_shape[1], cout << "C=" << XC << endl;							//The number of input channel
	XH = x_shape[2], cout << "H=" << XH << endl;							//The size of input height
	XW = x_shape[3], cout << "W=" << XW << endl;							//The size of input weight
	input_size = 1;
	for (int i = 0; i < 4; i++) input_size *= x_shape[i];

	FH = kernel_size, cout << "FH=" << FH << endl;							//The size of pooling height
	FW = kernel_size, cout << "FW=" << FW << endl;							//The size of pooling weight

	this->stride = stride, cout << "Stride=" << stride << endl;				//The number of stride

	OH = int(1.0 + (XH - FH) / stride), cout << "OH=" << OH << endl;		//The size of output height
	OW = int(1.0 + (XW - FW) / stride), cout << "OW=" << OW << endl;		//The size of output weight

	this->dev = dev;
	this->type = type;
	this->train_flg = train_flg;

	/*return data shape*/
	x_shape[0] = XN;
	x_shape[1] = XC;
	x_shape[2] = OH;
	x_shape[3] = OW;
	output_size = 1;
	for (int i = 0; i < 4; i++) output_size *= x_shape[i];
}

float* Pooling::forward_cpu(float* x) {

	float* col = NULL;
	col = stride_forward(x, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	col = transpose(col, XN, XC, FH, FW, OH, OW, 0, 4, 5, 1, 2, 3);

	float* out = NULL;
	if (type == "max") {
		if (train_flg == 1) out = max_poolingForward(argMax, col, XN*OH*OW*XC, FH*FW);
		else out = max_poolingForward(col, XN*OH*OW*XC, FH*FW);
	}
	else if (type == "avg") out = avg_poolingForward(col, XN*OH*OW*XC, FH*FW);

	out = transpose(out, XN, OH, OW, XC, 0, 3, 1, 2);

	return out;
}

float* Pooling::forward_gpu(float* x) {

	float* col = NULL;
	col = stride_forward_gpu(x, stride, XN, XC, FH, FW, OH, OW, XH, XW);
	col = transpose_gpu(col, XN, XC, FH, FW, OH, OW, 0, 4, 5, 1, 2, 3);

	float* out = NULL;
	if (type == "max") {
		if (train_flg == 1) out = max_poolingForward_gpu(argMax, col, XN*OH*OW*XC, FH*FW);
		else out = max_poolingForward_gpu(col, XN*OH*OW*XC, FH*FW);
	}
	else if (type == "avg") out = avg_poolingForward_gpu(col, XN*OH*OW*XC, FH*FW);

	out = transpose_gpu(out, XN, OH, OW, XC, 0, 3, 1, 2);

	return out;
}

float* Pooling::backward_cpu(float* dout) {
	
	dout = transpose(dout, XN, XC, OH, OW, 0, 2, 3, 1);

	float* dcol = NULL;
	if (type == "max") dcol = max_poolingBackward(argMax, dout, XN*OH*OW*XC, FH*FW);
	else if (type == "avg") dcol = avg_poolingBackward(dout, XN*OH*OW*XC, FH*FW);
	
	dcol = transpose(dcol, XN, OH, OW, XC, FH, FW, 0, 3, 4, 5, 1, 2);

	float* dx = NULL;
	dx = stride_backward(dcol, stride, XN, XC, FH, FW, OH, OW, XH, XW);

	return dx;

}

float* Pooling::backward_gpu(float* dout) {

	dout = transpose_gpu(dout, XN, XC, OH, OW, 0, 2, 3, 1);

	float* dcol = NULL;
	if (type == "max") dcol = max_poolingBackward_gpu(argMax, dout, XN*OH*OW*XC, FH*FW);
	else if (type == "avg") dcol = avg_poolingBackward_gpu(dout, XN*OH*OW*XC, FH*FW);

	dcol = transpose_gpu(dcol, XN, OH, OW, XC, FH, FW, 0, 3, 4, 5, 1, 2);

	float* dx = NULL;
	dx = stride_backward_gpu(dcol, stride, XN, XC, FH, FW, OH, OW, XH, XW);

	return dx;
}

Pooling::~Pooling()
{
}
