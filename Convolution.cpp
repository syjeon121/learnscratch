#include "Convolution.h"

/*normal distribution for He method*/
random_device rd_conv;
mt19937 gen_conv(rd_conv());
normal_distribution<float> distribution_conv(0, 1);


Convolution::Convolution()
{
	buffer = 0;
}
 
void Convolution::init(string name, int* x_shape, int num_output, int kernel_size, const int pad, const int stride, int dev,
	int train_flg, int previous_size) {

	this->name = name;
	cout << "name=" << name << endl;

	XN = x_shape[0], cout << "N=" << XN << endl;									//The number of input data
	XC = x_shape[1], cout << "C=" << XC << endl;									//The number of input channel
	XH = x_shape[2], cout << "H=" << XH << endl;									//The size of input height
	XW = x_shape[3], cout << "W=" << XW << endl;									//The size of input weight
	input_size = 1;
	for (int i = 0; i < 4; i++) input_size *= x_shape[i];

	FN = num_output,	cout << "FN=" << FN << endl;								//The number of filter (the number of output channel)
	FC = XC,			cout << "FC=" << FC << endl;								//The number of filter channel (the number of input channel)
	FH = kernel_size,	cout << "FH=" << FH << endl;								//The size of filter height
	FW = kernel_size,	cout << "FW=" << FW << endl;								//The size of filter weight

	this->pad = pad,		cout << "Pad=" << pad << endl;							//The number of pad
	this->stride = stride,	cout << "Stride=" << stride << endl;					//The number of stride

	OH = int(1.0 + (XH + 2.0*pad - FH) / stride), cout << "OH=" << OH << endl;		//The size of output height
	OW = int(1.0 + (XW + 2.0*pad - FW) / stride), cout << "OW=" << OW << endl;		//The size of output weight


	dXH = XH + 2 * pad + stride - 1;
	dXW = XW + 2 * pad + stride - 1;

	this->dev = dev;
	nodeNum = previous_size;	//for He method
	this->train_flg = train_flg;

	/*initialize variables*/
	init_size();
	switch (dev) {
	case 1:
		init_cpu();
		break;
	case 2:
		init_gpu();
		break;
	default:
		cout << "device error for init var function!(1:cpu, 2:gpu)" << endl;
	}

	/*return data shape*/
	x_shape[0] = XN;
	x_shape[1] = FN;
	x_shape[2] = OH;
	x_shape[3] = OW;
	output_size = 1;
	for (int i = 0; i < 4; i++) output_size *= x_shape[i];
}

void Convolution::init_size() {
	size_w = FN*FC*FH*FW;
	size_dw = size_w;
	size_b = FN;
	size_db = size_b;
}

void Convolution::init_cpu() {

	int idx;

	w = new float[size_w];
	for (int i = 0; i < FN; i++) {
		for (int j = 0; j < FC; j++) {
			for (int k = 0; k < FH; k++) {
				for (int l = 0; l < FW; l++) {
					idx = i*(FC*FH*FW) + j*(FH*FW) + k*(FW)+l;

					w[idx] = 1.0;

					/*He method sqrt(2/n)*/
					//w[idx] = distribution_conv(gen_conv) * sqrt(2.0 / (nodeNum + 1e-7));
				}
			}
		}
	}
	buffer += size_w * sizeof(float);

	b = new float[size_b];
	for (int i = 0; i < FN; i++) {
		b[i] = 0;
	}
	buffer += size_b * sizeof(float);

	if (train_flg == 1) {
		dw = new float[size_dw];
		memset(dw, 0, size_dw * sizeof(float));
		buffer += size_dw * sizeof(float);

		db = new float[size_db];
		memset(db, 0, size_db * sizeof(float));
		buffer += size_db * sizeof(float);
	}

	cout << "cpu buffer size used in convolutional layer : " << buffer / (1000 * 1000) << "(MB)" << endl << endl;

} 

void Convolution::init_gpu() {


	host_w = new float[size_w];
	memset(host_w, 0, size_w * sizeof(float));
	for (int i = 0; i < size_w; i++) {

		host_w[i] = 1.0;

		/*He method sqrt(2/n)*/
		//host_w[i] = distribution_conv(gen_conv)*sqrt(2.0 / (nodeNum + 1e-7));

		/*gaussian*/
		//host_w[i] = distribution_conv(gen_conv) * 0.01;	//caffe gaussian filler
	}
	gpuErrchk(cudaMalloc((void**)&w, size_w * sizeof(float)));
	gpuErrchk(cudaMemcpy(w, host_w, size_w * sizeof(float), cudaMemcpyHostToDevice));
	buffer += size_w * sizeof(float);

	host_b = new float[size_b];
	memset(host_b, 0, size_b * sizeof(float));
	for (int i = 0; i < size_b; i++) {
		host_b[i] = 0;
	}
	gpuErrchk(cudaMalloc((void**)&b, size_b * sizeof(float)));
	gpuErrchk(cudaMemcpy(b, host_b, size_b * sizeof(float), cudaMemcpyHostToDevice));
	buffer += size_b * sizeof(float);

	if (train_flg == 1) {
		gpuErrchk(cudaMalloc((void**)&dw, size_dw * sizeof(float)));
		gpuErrchk(cudaMemset(dw, 0, size_dw * sizeof(float)));
		buffer += size_dw * sizeof(float);

		gpuErrchk(cudaMalloc((void**)&db, size_db * sizeof(float)));
		gpuErrchk(cudaMemset(db, 0, size_db * sizeof(float)));
		buffer += size_db * sizeof(float);
	}

	cout << "gpu buffer size used in convolutional layer : " << buffer / (1000 * 1000) << "(MB)" << endl << endl;

}

void Convolution::init_trained_parameter(ifstream& fin_w, ifstream& fin_b)
{

	if (dev == 1)
	{
		if (fin_w.is_open()) {
			for (int i = 0; i < size_w; i++)
				fin_w >> w[i];
		}

		if (fin_b.is_open()) {
			for (int i = 0; i < size_b; i++)
				fin_b >> b[i];
		}
	}
	else if (dev == 2)
	{
		if (fin_w.is_open()) {
			for (int i = 0; i < size_w; i++) 
				fin_w >> host_w[i];
		}
		gpuErrchk(cudaMemcpy(w, host_w, size_w * sizeof(float), cudaMemcpyHostToDevice));



		if (fin_b.is_open()) {
			for (int i = 0; i < size_b; i++) 
				fin_b >> host_b[i];
		}
		gpuErrchk(cudaMemcpy(b, host_b, size_b * sizeof(float), cudaMemcpyHostToDevice));
	}
}

void Convolution::save_trained_parameter(ofstream& fout_w, ofstream& fout_b)
{
	if (dev == 1)
	{

	}
	else if (dev == 2)
	{
		gpuErrchk(cudaMemcpy(host_w, w, size_w * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < size_w; i++)
			fout_w << host_w[i] << endl;
		
		gpuErrchk(cudaMemcpy(host_b, b, size_b * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < size_b; i++)
			fout_b << host_b[i] << endl;
	}
}


float* Convolution::forward_cpu(float* x) {
	txt = 'f';

	if (pad) {	
		float* x_pad = NULL;
		x_pad = padding(x, pad, XN, XC, XH, XW);
		col = stride_forward(x_pad, stride, XN, XC, FH, FW, OH, OW, XH + 2 * pad, XW + 2 * pad);
	}
	else col = stride_forward(x, stride, XN, XC, FH, FW, OH, OW, XH, XW);

	col = transpose(col, XN, XC, FH, FW, OH, OW, 0, 4, 5, 1, 2, 3);
	w = transpose(w, FN, FC*FH*FW);

	float* out = NULL;
	out = dot(col, w, XN*OH*OW, FN, XC*FH*FW);
	if (train_flg == 2) delete_cpu<float>(col);

	sum_forward(out, b, XN*OH*OW, FN);
	out = transpose(out, XN, OH, OW, FN, 0, 3, 1, 2);

	return out;
}

float* Convolution::forward_gpu(float* x) {
	txt = 'f';

	if (pad)
	{
		float* x_pad = NULL;
		x_pad = padding_gpu(x, pad, XN, XC, XH, XW);
		col = stride_forward_gpu(x_pad, stride, XN, XC, FH, FW, OH, OW, XH + 2 * pad, XW + 2 * pad);

	}
	else col = stride_forward_gpu(x, stride, XN, XC, FH, FW, OH, OW, XH + 2 * pad, XW + 2 * pad);

	col = transpose_gpu(col, XN, XC, FH, FW, OH, OW, 0, 4, 5, 1, 2, 3);
	w = transpose_gpu(w, FN, FC*FH*FW);

	float* out = NULL;
	out = dot_gpu(col, w, XN*OH*OW, FN, XC*FH*FW);
	if (train_flg == 2) delete_gpu<float>(col);

	sum_forward_gpu(out, b, XN*OH*OW, FN);
	out = transpose_gpu(out, XN, OH, OW, FN, 0, 3, 1, 2);
	return out;
}

float* Convolution::backward_cpu(float* dout) {

	
	dout = transpose(dout, XN, FN, OH, OW, 0, 2, 3, 1);
	sum_backward(db, dout, XN*OH*OW, FN);

	col = transpose(col, XN*OH*OW, XC*FH*FW);
	_dot(dw, col, dout, FC*FH*FW, FN, XN*OH*OW);
	delete_cpu<float>(col);

	dw = transpose(dw, FC*FH*FW, FN);

	float* dcol = NULL;
	dcol = dot(dout, w, XN*OH*OW, FC*FH*FW, FN);
	delete_cpu<float>(dout);

	dcol = transpose(dcol, XN, OH, OW, XC, FH, FW, 0, 3, 4, 5, 1, 2);
	
	float* dx = NULL;
	dx = stride_backward(dcol, stride, XN, XC, FH, FW, OH, OW, dXH, dXW);

	if (pad) {
		dx = padding(dx, pad, XN, XC, XH, XW, stride);
	}

	return dx;
}

float* Convolution::backward_gpu(float* dout) {


	dout = transpose_gpu(dout, XN, FN, OH, OW, 0, 2, 3, 1);
	sum_backward_gpu(db, dout, XN*OH*OW, FN);

	col = transpose_gpu(col, XN*OH*OW, XC*FH*FW);
	_dot_gpu(dw, col, dout, FC*FH*FW, FN, XN*OH*OW);
	delete_gpu<float>(col);

	dw = transpose_gpu(dw, FC*FH*FW, FN);

	float* dcol = NULL;
	dcol = dot_gpu(dout, w, XN*OH*OW, FC*FH*FW, FN);
	delete_gpu<float>(dout);

	dcol = transpose_gpu(dcol, XN, OH, OW, XC, FH, FW, 0, 3, 4, 5, 1, 2);

	float* dx = NULL;
	dx = stride_backward_gpu(dcol, stride, XN, XC, FH, FW, OH, OW, dXH, dXW);

	if (pad) {
		dx = padding_gpu(dx, pad, XN, XC, XH, XW, stride);
	}

	return dx;

}

Convolution::~Convolution()
{
	if (dev == 1) {


		delete[] w, b;
		if (train_flg == 1) {
			delete[] dw, db;
		}
	}
	else if (dev == 2) {


		cudaFree(b);
		cudaFree(w);
		delete[] host_w, host_b;
		host_w = NULL, host_b = NULL;

		if (train_flg == 1) {
			cudaFree(dw);
			cudaFree(db);
		}
	}

	w = NULL, b = NULL;
	dw = NULL, db = NULL;
}
