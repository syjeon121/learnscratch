#include "Network.h"

Network::Network(){
	buffer_total = 0;
}

void Network::init(int* input_shape, int the_number_of_class, int dev, int train_flg)
{
	for(int i = 0; i < 4; i++) this->input_shape[i] = input_shape[i];

	this->dev = dev;
	this->train_flg = train_flg;

	init_layer();
	init_param();
}

void Network::init_layer()
{
	/*model: */

	
	the_number_of_conv = 0;
	the_number_of_fc = 0;

	for (int i = 0; i < the_number_of_conv; i++) {
		buffer_total += conv[i].buffer;
	}
	for (int i = 0; i < the_number_of_fc; i++) {
		buffer_total += fc[i].buffer;
	}
	cout << "total buffer : " << buffer_total / (1000 * 1000) << "(MB)" << endl;
}

void Network::init_param()
{
	int the_number_of_weight_layer = the_number_of_conv + the_number_of_fc;

	parameter_shape = new int[the_number_of_weight_layer * 2];		//weight and bias
	for (int n = 0; n < the_number_of_conv; n++)
	{
		parameter_shape[n*2 + 0] = conv[n].size_w;
		parameter_shape[n*2 + 1] = conv[n].size_b;
	}
	for (int n = 0; n < the_number_of_fc; n++)
	{
		parameter_shape[(the_number_of_conv + n) * 2 + 0] = fc[n].size_w;
		parameter_shape[(the_number_of_conv + n) * 2 + 1] = fc[n].size_b;
	}

	parameter = new float*[the_number_of_weight_layer * 2];
	gradient = new float*[the_number_of_weight_layer * 2];
	for (int n = 0; n < the_number_of_conv; n++)
	{
		parameter[n * 2 + 0] = conv[n].w;
		parameter[n * 2 + 1] = conv[n].b;
	}
	for (int n = 0; n < the_number_of_fc; n++)
	{
		parameter[(the_number_of_conv + n) * 2 + 0] = fc[n].w;
		parameter[(the_number_of_conv + n) * 2 + 1] = fc[n].b;
	}

	for (int n = 0; n < the_number_of_conv; n++)
	{
		gradient[n * 2 + 0] = conv[n].dw;
		gradient[n * 2 + 1] = conv[n].db;
	}
	for (int n = 0; n < the_number_of_fc; n++)
	{
		gradient[(the_number_of_conv + n) * 2 + 0] = fc[n].dw;
		gradient[(the_number_of_conv + n) * 2 + 1] = fc[n].db;
	}

}


float* Network::predict_cpu(float* x) {

	return 0;
}

float* Network::predict_gpu(float* x) {


	return 0;
}


float Network::loss_cpu(float* x, int* t) {

	return 0;
}

float Network::loss_gpu(float* x, int* t) {
	
	return 0;
}


float Network::gradient_cpu(float* x, int* t) {


	return 0;
}

float Network::gradient_gpu(float* x, int* t) {

	return 0;
}


void Network::save_parameter_cpu()
{

}

void Network::save_parameter_gpu()
{

}

Network::~Network()
{
	delete[] parameter_shape;
	parameter_shape = NULL;
}