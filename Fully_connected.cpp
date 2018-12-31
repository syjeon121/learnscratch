#include "Fully_connected.h"

/*normal distribution for He method*/
random_device rd_fc;
mt19937 gen_fc(rd_fc());
normal_distribution<float> distribution_fc(0, 1);

Fully_connected::Fully_connected() {
	buffer = 0;
}

void Fully_connected::init(string name, int dim, int* x_shape, int num_output, int dev, 
	int train_flg, int previous_size) {

	this->name = name;
	cout << endl << "name=" << name << endl;

	if (dim == 2) {
		XN = x_shape[0], cout << "N=" << XN << endl;									
		IN = x_shape[1], cout << "I=" << IN << endl;									
	}
	else if (dim == 4) {
		XN = x_shape[0], cout << "N=" << XN << endl;			//The number of input data
		XC = x_shape[1], cout << "C=" << XC << endl;			//The number of input channel
		XH = x_shape[2], cout << "H=" << XH << endl;			//The size of input height
		XW = x_shape[3], cout << "W=" << XW << endl;			//The size of input weight
		IN = XC*XH*XW;
	}
	input_size = 1;
	for (int i = 0; i < dim; i++) input_size *= x_shape[i];


	WN = num_output, cout << "WN=" << WN << endl;					//The number of weight

	this->dev = dev;
	nodeNum = previous_size;
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
	x_shape[1] = WN;
	output_size = 1;
	for (int i = 0; i < 2; i++) output_size *= x_shape[i];

}

void Fully_connected::init_size() {
	size_w = IN*WN;
	size_dw = size_w;
	size_b = WN;
	size_db = size_b;
}

void Fully_connected::init_cpu() {

	int idx;

	w = new float[size_w];
	for (int i = 0; i < IN; i++) {
		for (int j = 0; j < WN; j++) {

			idx = i*WN + j;

			w[idx] = 1.0;

			/*He method sqrt(2/n)*/
			//w[idx] = distribution_conv(gen_conv) * sqrt(2.0 / (nodeNum + 1e-7));

		}
	}
	buffer += size_w * sizeof(float);

	b = new float[size_b];
	for (int i = 0; i < WN; i++) {
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

	cout << "cpu buffer size used in fully connected layer : " << buffer / (1000 * 1000) << "(MB)" << endl << endl;

}

void Fully_connected::init_gpu() {
	host_w = new float[size_w];
	memset(host_w, 0, size_w * sizeof(float));
	for (int i = 0; i < size_w; i++) {

		host_w[i] = 1.0;

		/*He method sqrt(2/n)*/
		//host_w[i] = distribution_conv(gen_conv)*sqrt(2.0 / nodeNum);

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

	cout << "gpu buffer size used in fully connected layer : " << buffer / (1000 * 1000) << "(MB)" << endl << endl;
}

void Fully_connected::init_trained_parameter(ifstream& fin_w, ifstream& fin_b)
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

void Fully_connected::save_trained_parameter(ofstream& fout_w, ofstream& fout_b)
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

float* Fully_connected::forward_cpu(float* x) {

	float* out = NULL;
	out = dot(x, w, XN, WN, IN);

	if (train_flg == 1) this->x = x;
	else if (train_flg == 2) delete_cpu<float>(x);

	sum_forward(out, b, XN, WN);

	return out;
}

float* Fully_connected::forward_gpu(float* x) {
	float* out = NULL;
	out = dot_gpu(x, w, XN, WN, IN);

	if (train_flg == 1) this->x = x;
	else if (train_flg == 2) delete_gpu<float>(x);

	sum_forward_gpu(out, b, XN, WN);

	return out;
}

float* Fully_connected::backward_cpu(float* dout) {

	x = transpose(x, XN, IN);

	_dot(dw, x, dout, IN, WN, XN); 
	sum_backward(db, dout, XN, WN);

	w = transpose(w, IN, WN);
	float* dx = NULL;
	dx = dot(dout, w, XN, IN, WN);
	delete_cpu<float>(dout);

	w = transpose(w, WN, IN);
	return dx;
}

float* Fully_connected::backward_gpu(float* dout) {
	x = transpose_gpu(x, XN, IN);

	_dot_gpu(dw, x, dout, IN, WN, XN);
	sum_backward_gpu(db, dout, XN, WN);

	w = transpose_gpu(w, IN, WN);
	float* dx = NULL;
	dx = dot_gpu(dout, w, XN, IN, WN);
	delete_gpu<float>(dout);

	w = transpose_gpu(w, WN, IN);
	return dx;
}

Fully_connected::~Fully_connected()
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







