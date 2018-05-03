#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
//#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
#include <random>
#include <ctime>
//#include <random>
//#include "CImg\CImg.h"

#define iterations 25000

//using namespace cimg_library;

typedef struct {
	double** matrix;
	int h, w;
} Matrix;

typedef struct{
    double** matrix;
    int top,bot,left,right;
} matrix;

int num_layers = 0;
Matrix* layers;//array of 2D arrays
Matrix* v;//array of layers potential field (Y[i-1]*W[i])
Matrix* activated;//array of activated layers
Matrix* conv_coord;//matrix with nulles and maxes
Matrix* conv_cores;
Matrix* dw;
double*** errors;//array of all errors, including hidden layers
double err;
double sigma;
int step = 0;

Matrix* x_train = NULL;
Matrix* y_train = NULL;

typedef unsigned __int16 WORD;

typedef struct {
	WORD   bfType;         // 0x4d42 | 0x4349 | 0x5450
	int    bfSize;         // размер файла
	int    bfReserved;     // 0
	int    bfOffBits;      // смещение до поля данных,
						   // обычно 54 = 16 + biSize
	int    biSize;         // размер струкуры в байтах:
						   // 40(BITMAPINFOHEADER) или 108(BITMAPV4HEADER)
						   // или 124(BITMAPV5HEADER)
	int    biWidth;        // ширина в точках
	int    biHeight;       // высота в точках
	WORD   biPlanes;       // всегда должно быть 1
	WORD   biBitCount;     // 0 | 1 | 4 | 8 | 16 | 24 | 32
	int    biCompression;  // BI_RGB | BI_RLE8 | BI_RLE4 |
						   // BI_BITFIELDS | BI_JPEG | BI_PNG
						   // реально используется лишь BI_RGB
	int    biSizeImage;    // Количество байт в поле данных
						   // Обычно устанавливается в 0
	int    biXPelsPerMeter;// горизонтальное разрешение, точек на дюйм
	int    biYPelsPerMeter;// вертикальное разрешение, точек на дюйм
	int    biClrUsed;      // Количество используемых цветов
						   // (если есть таблица цветов)
	int    biClrImportant; // Количество существенных цветов.
						   // Можно считать, просто 0
} BMPheader;



double max(double a, double b) {
	return (a > b) ? a : b;
}

double get_max(int *len, ...) {
	++len;
	double max = *len;
	while(*len) {
		if (*len > max)
			max = *len;
	}
	return max;
}

Matrix Transpose(Matrix a) {
	Matrix res;
	res.matrix = (double**)malloc(a.w * sizeof(double*));
	for (int i = 0; i < a.w; i++) {
		res.matrix[i] = (double*)malloc(a.h * sizeof(double));
		for (int j = 0; j < a.h; j++)
			res.matrix[i][j] = a.matrix[j][i];
	}
	res.h = a.w;
	res.w = a.h;
	return res;
}

Matrix Multiply1T(Matrix a, Matrix b) {
	if (a.h != b.h) {
		double* a = NULL;
		int b = 5 + *a;//raise error
	}
	else
	{
		double **mx = (double**)malloc(a.w * sizeof(double*));
		for (int i = 0; i < a.w; i++) {
			mx[i] = (double*)malloc(b.w * sizeof(double));
			for (int j = 0; j < b.w; j++)
			{
				mx[i][j] = 0;
				for (int k = 0; k < a.h; k++)
					mx[i][j] += a.matrix[k][i] * b.matrix[k][j];
			}
		}
		Matrix res;;
		res.matrix = mx;
		res.h = a.w;
		res.w = b.w;
		return res;
	}
}

Matrix Multiply2T(Matrix a, Matrix b) {
	if (a.w != b.w) {
		double* a = NULL;
		int b = 5 + *a;//raise error
	}
	else
	{
		double **mx = (double**)malloc(a.h * sizeof(double*));
		for (int i = 0; i < a.h; i++) {
			mx[i] = (double*)malloc(b.h * sizeof(double));
			for (int j = 0; j < b.h; j++)
			{
				mx[i][j] = 0;
				for (int k = 0; k < a.w; k++)
					mx[i][j] += a.matrix[i][k] * b.matrix[j][k];
			}
		}
		Matrix res;;
		res.matrix = mx;
		res.h = a.h;
		res.w = b.h;
		return res;
	}
}

Matrix MultiplyAllT(Matrix a, Matrix b) {
	if (a.h != b.w) {
		double* a = NULL;
		int b = 5 + *a;//raise error
	}
	else
	{
		double **mx = (double**)malloc(a.h * sizeof(double*));
		for (int i = 0; i < a.h; i++) {
			mx[i] = (double*)malloc(b.h * sizeof(double));
			for (int j = 0; j < b.h; j++)
			{
				mx[i][j] = 0;
				for (int k = 0; k < a.h; k++)
					mx[i][j] += a.matrix[k][i] * b.matrix[j][k];
			}
		}
		Matrix res;;
		res.matrix = mx;
		res.h = a.w;
		res.w = b.h;
		return res;
	}
}

Matrix Multiply(Matrix a, Matrix b) {
	if (a.w != b.h) {
		double* a = NULL;
		int b = 5 + *a;//raise error
	}
	else
	{
		double **matrix = (double**)malloc(a.h * sizeof(double*));
		for (int i = 0; i < a.h; i++) {
			matrix[i] = (double*)malloc(b.w * sizeof(double));
			for (int j = 0; j < b.w; j++)
			{
				matrix[i][j] = 0;
				for (int k = 0; k < a.w; k++)
					matrix[i][j] += a.matrix[i][k] * b.matrix[k][j];
			}
		}
		Matrix mx;
		mx.matrix = matrix;
		mx.h = a.h;
		mx.w = b.w;
		return mx;
	}
}

Matrix MultiplyD(Matrix a, double b) {
	double** res = (double**)malloc(a.h * sizeof(double*));
	for (int i = 0; i < a.h; i++) {
		res[i] = (double*)malloc(a.w * sizeof(double));
		for (int j = 0; j < a.w; j++)
			res[i][j] = a.matrix[i][j] * b;
	}
	Matrix mx;
	mx.matrix = res;
	mx.h = a.h;
	mx.w = a.w;
	return mx;
}

Matrix Dot(Matrix a, Matrix b) {
	double** res = (double**)malloc(a.h * sizeof(double*));
	for (int i = 0; i < a.h; i++) {
		res[i] = (double*)malloc(a.w * sizeof(double));
		for (int j = 0; j < a.w; j++)
			res[i][j] = a.matrix[i][j] * b.matrix[i][j];
	}
	Matrix mx;
	mx.matrix = res;
	mx.h = a.h;
	mx.w = a.w;
	return mx;
}

Matrix Minus(Matrix a, Matrix b) {
	double** res = (double**)malloc(a.h * sizeof(double*));
	for (int i = 0; i < a.h; i++) {
		res[i] = (double*)malloc(a.w * sizeof(double));
		for (int j = 0; j < a.w; j++)
			res[i][j] = a.matrix[i][j] - b.matrix[i][j];
	}
	Matrix mx;
	mx.matrix = res;
	mx.h = a.h;
	mx.w = a.w;
	return mx;
}

void Plused(Matrix a, Matrix b) {
	for (int i = 0; i < a.h; i++) {
		for (int j = 0; j < a.w; j++)
			a.matrix[i][j] += b.matrix[i][j];
	}
}

double Sigmoid(double x) {
	return 1 / (1 + exp(-1 * x));
}

//Объеденить сигмоиду и её производную в одну функцию:
Matrix getActivationSigmoid(Matrix x) {
	double** res = (double**)malloc(x.h * sizeof(double*));
	for (int i = 0; i < x.h; i++) {
		res[i] = (double*)malloc(x.w * sizeof(double));
		for (int j = 0; j < x.w; j++)
			res[i][j] = 1 / (1 + exp(-1 * x.matrix[i][j]));
	}
	Matrix mx;
	mx.matrix = res;
	mx.h = x.h;
	mx.w = x.w;
	return mx;
}

Matrix getSigmoid_d(Matrix x) {
	double** res = (double**)malloc(x.h * sizeof(double*));
	for (int i = 0; i < x.h; i++) {
		res[i] = (double*)malloc(x.w * sizeof(double));
		for (int j = 0; j < x.w; j++)
			res[i][j] = x.matrix[i][j] * (1 - x.matrix[i][j]);
	}
	Matrix mx;
	mx.matrix = res;
	mx.h = x.h;
	mx.w = x.w;
	return mx;
}

double** getActivationReLU(Matrix x) {
	double** res = (double**)malloc(x.h * sizeof(double*));
	for (int i = 0; i < x.h; i++) {
		res[i] = (double*)malloc(x.w * sizeof(double));
		for (int j = 0; j < x.w; j++)
			res[i][j] = max(0, x.matrix[i][j]);
	}
	return res;
}

double** getActivationAnalytic(Matrix x) {
	double** res = (double**)malloc(x.h * sizeof(double*));
	for (int i = 0; i < x.h; i++) {
		res[i] = (double*)malloc(x.w * sizeof(double));
		for (int j = 0; j < x.w; j++)
			res[i][j] = log10(1 + exp(x.matrix[i][j]));
	}
	return res;
}

double** getActivationAnalytic_d(double** x, int h, int w) {
	double** res = (double**)malloc(h * sizeof(double*));
	for (int i = 0; i < h; i++) {
		res[i] = (double*)malloc(w * sizeof(double));
		for (int j = 0; j < w; j++)
			res[i][j] = 1.0 / (1.0 + exp(-1 * x[i][j]));
	}
	return res;
}

double mean_squere(double y, double t) {
	return pow((t - y), 2) / 2;
}

//Make that
/*double** getActivationSoftmax(Matrix* x) {
	double** res = (double**)malloc(x->h * sizeof(double*));
	double sum_e = 0;
	for (int i = 0; i < x->h; i++) {
		res[i] = (double*)malloc(x->w * sizeof(double));
		for (int j = 0; j < x->w; j++) {
			sum_e += exp(x->matrix[i][j]);
		}
	}
	return res;
}*/

double** convolute(double** image, int i_w, int i_h, double** kernel, int k_w, int k_h, int step){
	matrix ker = { kernel, 0, k_h - 1, 0, k_w - 1 };
	matrix img = { image, 0, i_h - 1, 0, i_w - 1 };

    double** convoluted = (double**)malloc((i_h - step) * sizeof(double*));
    for(int i = 0; i < i_h - step; i++)
        convoluted[i] = (double*)malloc((i_w - step) * sizeof(double));

    //for convoluted matrix:
    int x = 0;
    int y = 0;
    while(ker.bot <= img.bot){
        while(ker.right <= img.right){
            double res = 0;
            for(int i = max(0, ker.left); i <= ker.right; i++){
                for(int j = max(0, ker.top); j <= ker.bot; j++){
                    res += img.matrix[i][j] * ker.matrix[i - ker.left][j - ker.top];
                }
            }
			//res /= k_w * k_h;
            convoluted[y][x] = res;
            ker.left += step;
            ker.right += step;
			++x;
        }
		++y;
		x = 0;
		ker.left = 0;
		ker.right = k_w - 1;
        ker.top += step;
        ker.bot += step;
    }
    return convoluted;
}

//refacore that:
Matrix max_pooling(Matrix x, int size, int n) {
	conv_coord[n].h = x.h;
	conv_coord[n].w = x.w;
	conv_coord[n].matrix = (double**)malloc(x.h * sizeof(double*));
	for (int i = 0; i < x.h; i++) {
		conv_coord[n].matrix[i] = (double*)malloc(x.w * sizeof(double));
		for (int j = 0; j < x.w; j++)
			conv_coord[n].matrix[i][j] = 0;
	}

	double** res = (double**)malloc((x.h / size) * sizeof(double*));
	for (int i = 0; i < x.h / size; i++)
		res[i] = (double*)malloc((x.w / size) * sizeof(double));

	int w = x.w;
	int h = x.h;
	int a = 0;
	int b = 0;
	int yc;
	int xc = yc = 0;
	for (int i = 0; i < h; i+=size) {
		for (int j = 0; j < w; j+=size) {
			double max = x.matrix[i][j];
			for (int q = 0; q < size; q++) {
				for (int w = 0; w < size; w++)
					if (x.matrix[i + q][j + w] > max) {
						max = x.matrix[i + q][j + w];
						xc = i + q;
						yc = j + w;
					}
			}
			res[a][b++] = max;
			conv_coord[n].matrix[xc][yc] = max;
		}
		++a;
		b = 0;
	}
	
	Matrix m;
	m.h = x.h / size;
	m.w = x.w / size;
	m.matrix = res;
	return m;
}

void link(Matrix x, Matrix kernel, int cell) {
	int hi = (cell - 1) / kernel.w;
	int wi = (cell - 1) % kernel.w;
	int d = kernel.h - hi - 1;
	for (int h = hi; h < x.h - d; h++) {
		int ind = wi + h * x.w;
		for (int i = 0; i < x.w - kernel.w + 1; i++) {
			printf("%f\n", x.matrix[0][i + ind]);
		}
	}
}

Matrix reshape(Matrix x, int h, int w) {
	if (x.w*x.h != h * w) {
		int* a = NULL;
		a + 5;
	}
	Matrix out;
	out.h = h;
	out.w = w;

	out.matrix = (double**)malloc(h * sizeof(double*));
	//so many iterations:
	for (int i = 0; i < h; i++) {
		out.matrix[i] = (double*)malloc(w * sizeof(double));
	}
	for (int i = 0; i < x.h; i++) {
		for (int j = 0; j < x.w; j++) {
			int linear_x = i * x.w + j;
			out.matrix[linear_x / w][linear_x%w] = x.matrix[i][j];
		}
	}
	return out;
}

void print(double** m, int h, int w) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++)
			printf("%f ", m[i][j]);
		printf("\n");
	}
}

void print(Matrix m) {
	for (int i = 0; i < m.h; i++) {
		for (int j = 0; j < m.w; j++)
			printf("%f ", m.matrix[i][j]);
		printf("\n");
	}
}

void clear(Matrix x) {
	for (int i = 0; i < x.h; i++)
		free(x.matrix[i]);
	free(x.matrix);
}

Matrix predict(Matrix x) {
	v[0] = Multiply(layers[0], x);
	activated[0] = getActivationSigmoid(v[0]);
	for (int i = 1; i < num_layers; i++) {
		v[i] = Multiply(layers[i], activated[i - 1]);
		activated[i] = getActivationSigmoid(v[i]);
	}
	return activated[num_layers - 1];
}

Matrix predictConv(Matrix x) {
	conv_coord = new Matrix;
	conv_coord[0].h = 2;
	conv_coord[0].w = 2;
	conv_coord[0].matrix = new double*[2];
	for (int i = 0; i < 2; i++) {
		conv_coord[0].matrix[i] = new double[2];
		for (int j = 0; j < 2; j++)
			conv_coord[0].matrix[i][j] = 1;
	}

	Matrix cnv;
	cnv.matrix = convolute(x.matrix, x.w, x.h, conv_coord[0].matrix, conv_coord[0].w, conv_coord[0].h, 1);
	cnv.h = x.h - 1;
	cnv.w = x.w - 1;
	Matrix input = reshape(cnv, cnv.h*cnv.w, 1);
	return predict(input);
}

//make inertia
double backpropagation(Matrix x, Matrix t, double speed) {
	Matrix y = predictConv(x);
	
	Matrix error = Minus(t, y);

	Matrix* d_v = (Matrix*)malloc(num_layers * sizeof(Matrix));
	for (int i = 0; i < num_layers; i++) {
		d_v[i] = getSigmoid_d(activated[i]);
	}

	Matrix* sigma = (Matrix*)malloc(num_layers * sizeof(Matrix));

	//calculate all sigma's
	sigma[num_layers - 1] = Dot(error, d_v[num_layers - 1]);
	for (int i = num_layers - 1; i > 0; i--) {
		Matrix ws = Multiply1T(layers[i], sigma[i]);
		sigma[i - 1] = Dot(ws, d_v[i - 1]);
		clear(ws);
	}
	Matrix sigma_conv;
	
	//calculate all deltaW and update weights without last
	for (int i = num_layers - 1; i > 0; i--) {
		Matrix dw = Multiply2T(sigma[i], activated[i - 1]);
		Matrix dws = MultiplyD(dw, speed);
		Plused(layers[i], dws);
		clear(dw);
		clear(dws);
	}

	Matrix dw1 = Multiply2T(sigma[0], x);
	Matrix dw = MultiplyD(dw1, speed);
	Plused(layers[0], dw);
	clear(dw);
	clear(dw1);


	for (int i = 0; i < num_layers; i++) {
		clear(sigma[i]);
		clear(d_v[i]);
	}
	free(sigma);
	free(d_v);

	double mse = 0;
	for (int i = 0; i < t.w; i++)
		mse += error.matrix[0][i] * error.matrix[0][i];
	mse *= 0.5;
	clear(error);
	for (int i = 0; i < num_layers; i++) {
		clear(v[i]);
		clear(activated[i]);
	}
	return mse;
}

double random() {
	double k = 0;
	int n = 12;
	for (int i = 0; i < n; i++)
		k += ((rand() % 1000) / 1000.0) - 0.5;
	k /= n;
	return 0;
}

//make variable number of var
//layers get random number in [-1; 1]. Make random generator with regular distrib
void InitNN(int n, int x1, int x2, int x3, int o) {
	num_layers = n;
	srand(time(NULL));
	v = (Matrix*)malloc(n * sizeof(Matrix));
	activated = (Matrix*)malloc(n * sizeof(Matrix));
	layers = (Matrix*)malloc(n * sizeof(Matrix));
	dw = (Matrix*)malloc(n*sizeof(Matrix));
	/*for (int i = 0; i < n; i++) {
		layers[i].matrix = 
	}*/

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.5);

	layers[0].matrix = (double**)malloc(x2 * sizeof(double*));
	for (int i = 0; i < x2; i++) {
		layers[0].matrix[i] = (double*)malloc(x1 * sizeof(double));
		for (int j = 0; j < x1; j++)
			layers[0].matrix[i][j] = distribution(generator);
	}
	layers[0].h = x2;
	layers[0].w = x1;
	
	layers[1].matrix = (double**)malloc(x3 * sizeof(double*));
	for (int i = 0; i < x3; i++) {
		layers[1].matrix[i] = (double*)malloc(x2 * sizeof(double));
		for (int j = 0; j < x2; j++)
			layers[1].matrix[i][j] = distribution(generator);
	}
	layers[1].h = x3;
	layers[1].w = x2;

	layers[2].matrix = (double**)malloc(o * sizeof(double*));
	for (int i = 0; i < o; i++) {
		layers[2].matrix[i] = (double*)malloc(x3 * sizeof(double));
		for (int j = 0; j < x3; j++)
			layers[2].matrix[i][j] = distribution(generator);
	}	
	layers[2].h = o;
	layers[2].w = x3;
}

void print_weights() {
	printf("Number of layers: %d\n", num_layers);
	for (int i = 0; i < num_layers; i++) {
		printf("Height and width: %d %d\n", layers[i].h, layers[i].w);
		for (int j = 0; j < layers[i].h; j++) {
			for (int k = 0; k < layers[i].w; k++)
				printf("%f ", layers[i].matrix[j][k]);
		printf("\n");
		}
		printf("\n------------------------------\n");
	}
}

void save_weights(FILE* file) {
	fprintf(file, "%d\n", num_layers);
	for (int i = 0; i < num_layers; i++) {
		fprintf(file, "%d\n%d\n", layers[i].h, layers[i].w);
		for (int j = 0; j < layers[i].h; j++) {
			for (int k = 0; k < layers[i].w; k++)
				fprintf(file, "%f\n", layers[i].matrix[j][k]);
		}
	}
}

void load_weights(FILE* file) {
	fscanf(file, "%d", &num_layers);
	for (int i = 0; i < num_layers; i++) {
		fscanf(file, "%d", layers[i].h);
		fscanf(file, "%d", layers[i].w);
		for (int j = 0; j < layers[i].h; j++) {
			for (int k = 0; k < layers[i].w; k++)
				fscanf(file, "%f", layers[i].matrix[j][k]);
		}
	}
}

double chtod(char* b) {
	double res = 0;
	if (b[0] == '-')
		return -1;
	for (int i = 0; b[i] != 0 && b[i] != '\n'; i++) {
		res += b[i] - 48;
		res *= 10;
	}
	return res / 10;
}

#define wi 352
#define hi 143

char* read(FILE* file) {
	char* buf = (char*)malloc(4*sizeof(char));
	for (int i = 0; i < 4; i++)
		buf[i] = 0;
	char c = fgetc(file);
	int i = 0;
	while (c != ',' && c != '\n'&&c != '\0' && c!= -1) {
		buf[i] = c;
		c = fgetc(file);
	}
	return buf;
}


int main1()
{
	const int nrolls = 10000;  // number of experiments
	const int nstars = 100;    // maximum number of stars to distribute

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(5.0, 2.0);

	int p[10] = {};

	for (int i = 0; i<nrolls; ++i) {
		double number = distribution(generator);
		if ((number >= 0.0) && (number<10.0)) ++p[int(number)];
	}

	std::cout << "normal_distribution (5.0,2.0):" << std::endl;

	for (int i = 0; i<10; ++i) {
		std::cout << i << "-" << (i + 1) << ": ";
		std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
	}

	return 0;
}

void get_data(int n) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.09);
	
	static int k = 0;
	x_train = new Matrix;
	y_train = new Matrix;

	x_train[0].h = 64;
	x_train[0].w = 1;
	y_train[0].h = 1;
	y_train[0].w = 1;
	x_train[0].matrix = (double**)malloc(64 * sizeof(double*));
	for (int i = 0; i < 64; i++) {
		x_train[0].matrix[i] = (double*)malloc(sizeof(double));
		x_train[0].matrix[i][0] = distribution(generator);
	}
	y_train[0].matrix = (double**)malloc(sizeof(double*));
	y_train[0].matrix[0] = (double*)malloc(sizeof(double));
	y_train[0].matrix[0][0] = 0;
	if (k++%n)
		return;

	x_train[0].matrix[9][0] = 1 + distribution(generator);
	x_train[0].matrix[10][0] = 1 + distribution(generator);
	x_train[0].matrix[13][0] = 1 + distribution(generator);
	x_train[0].matrix[14][0] = 1 + distribution(generator);
	x_train[0].matrix[17][0] = 1 + distribution(generator);
	x_train[0].matrix[18][0] = 1 + distribution(generator);
	x_train[0].matrix[21][0] = 1 + distribution(generator);
	x_train[0].matrix[22][0] = 1 + distribution(generator);
	x_train[0].matrix[33][0] = 1 + distribution(generator);
	x_train[0].matrix[38][0] = 1 + distribution(generator);
	x_train[0].matrix[42][0] = 1 + distribution(generator);
	x_train[0].matrix[43][0] = 1 + distribution(generator);
	x_train[0].matrix[44][0] = 1 + distribution(generator);
	x_train[0].matrix[45][0] = 1 + distribution(generator);

	y_train[0].matrix[0][0] = 1;
}

void print_input(Matrix m, int n) {
	printf("---------------------------------------------------\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			printf("%f ", m.matrix[i*n + j][0]);
		printf("\n");
	}
	printf("---------------------------------------------------\n");
}

#define train_size 6000
#define epochs 10

void main() {
	/*Matrix fir;
	fir.matrix = (double**)malloc(3 * sizeof(double*));
	for (int i = 0; i < 3; i++)
		fir.matrix[i] = (double*)malloc(2 * sizeof(double));
	fir.matrix[0][0] = 1;
	fir.matrix[0][1] = 2;
	fir.matrix[1][0] = 3;
	fir.matrix[1][1] = 4;
	fir.matrix[2][0] = 5;
	fir.matrix[2][1] = 6;
	fir.h = 3;
	fir.w = 2;
	Matrix s;
	s.matrix = (double**)malloc(3 * sizeof(double*));
	for (int i = 0; i < 3; i++)
		s.matrix[i] = (double*)malloc(2 * sizeof(double));
	s.matrix[0][0] = 1;
	s.matrix[0][1] = 2;
	s.matrix[1][0] = 3;
	s.matrix[1][1] = 4;
	s.matrix[2][0] = 5;
	s.matrix[2][1] = 6;
	s.h = 3;
	s.w = 2;

	
	Matrix th = Multiply2T(fir, s);
	print(th.matrix, th.h, th.w);

	FILE* mnist = fopen("D:\\ \\c++\\Convolution\\Convolution\\mnist_train.csv", "r");

	
	double** y = (double**)malloc(1000 * sizeof(double*));
	for (int i = 0; i < 1000; i++) {
		y[i] = (double*)malloc(10 * sizeof(double));
		for (int j = 0; j < 10; j++)
			y[i][j] = 0;
	}
	double** x = (double**)malloc(1000 * sizeof(double*));
	for (int i = 0; i < 1000; i++)
		x[i] = (double*)malloc(784 * sizeof(double));

	//double bufd = 0;
	char* bufc;
	for (int i = 0; i < 1000; i++) {
		//fscanf(mnist, "%f", &bufd);
		bufc = read(mnist);
		int b = chtod(bufc);
		free(bufc);
		if (b == -1)
			break;
		y[i][b] = 1;
		for (int j = 0; j < 784; j++) {
			bufc = read(mnist);
			b = chtod(bufc);
			free(bufc);
			if (b == -1)
				break;
			x[i][j] = b;
		}
	}
	fclose(mnist);

	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 784; j++) {
			if (x[i][j] < 0 || x[i][j] > 255)
				printf("Bugged: %d, %d\n", i, j);
		}
	}

	InitNN(3, 784, 392, 196, 10);
	FILE* out = fopen("D:\\ \\c++\\Convolution\\Convolution\\out.txt", "w");
	FILE* weights = fopen("D:\\ \\c++\\Convolution\\Convolution\\weights.txt", "w");
	for (int k = 0; k < 10; k++) {
		for (int i = 0; i < 1000; i++) {
			double err = backpropagation(x[i], y[i], 784, 392, 196, 10, 0.1);
			fprintf(out, "%f\n", err);
		}
		printf("%d\n", k);
	}
	save_weights(weights);
	fclose(out);
	fclose(weights);
	exit(0);

	FILE* f = fopen("D:\\ \\c++\\Convolution\\Convolution\\arr.txt", "r");
	char* buf = (char*)malloc(4 * sizeof(char));
	double** arr_img = (double**)malloc(hi * sizeof(double*));
	for (int i = 0; i < hi; i++) {
		arr_img[i] = (double*)malloc(wi * sizeof(double));
		for (int j = 0; j < wi; j++) {
			fgets(buf, 4, f);
			if (buf[0] == '\n') {
				if (j == 0) {
					--i;
					j = wi - 1;
				}
				else
					--j;
				continue;
			}
			arr_img[i][j] = chtod(buf);
		}
	}
	fclose(f);

	double** kern = (double**)malloc(3 * sizeof(double*));
	for (int i = 0; i < 3; i++)
		kern[i] = (double*)malloc(3 * sizeof(double));

	kern[0][0] = 0.5;
	kern[0][1] = 1;
	kern[0][2] = 0.5;
	kern[1][0] = 0.25;
	kern[1][1] = 1;
	kern[1][2] = 0.25;
	kern[2][0] = 0.75;
	kern[2][1] = 0.1;
	kern[2][2] = 0.75;

	//print(arr_img, hi, wi);

	double** conved = convolute(arr_img, hi, wi, kern, 3, 3, 3);
	/*int size = (hi - 1)*(wi - 1);
	//InitNN(size, 3, 3, 2);
	double** input_vec = (double**)malloc(sizeof(double*));
	input_vec[0] = (double*)malloc(size * sizeof(double));
	for (int i = 0; i < size; i++)
		input_vec[0][i] = conved[i / hi][i%wi];

	print(predict(input_vec, size), 1, 2);

	//print(conved, hi - 20, wi - 20);

 	FILE* f_out = fopen("D:\\ \\c++\\Convolution\\Convolution\\brr.txt", "w");
	for (int i = 0; i < wi - 4; i++) {
		for (int j = 0; j < hi - 4; j++) {
			fprintf(f_out, "%f\n", conved[i][j]);
		}
	}
	fclose(f_out);
	exit(0);

	double** inp = (double**)malloc(1 * sizeof(double*));
	inp[0] = (double*)malloc(5 * sizeof(double));
	//predict(inp, 5);

	/*double* y = (double*)malloc(2 * sizeof(double));
	double* t = (double*)malloc(2 * sizeof(double));

	y[0] = 0.6;
	y[1] = 0.45;
	t[0] = 1;
	t[1] = 0;

	backpropagation(y, t, 5, 3, 3, 2, 0.05);


	double** img = (double**)malloc(4 * sizeof(double*));
	for (int i = 0; i < 4; i++)
		img[i] = (double*)malloc(4 * sizeof(double));

	double** ker = (double**)malloc(2 * sizeof(double*));
	for (int i = 0; i < 2; i++)
		ker[i] = (double*)malloc(2 * sizeof(double));

	img[0][0] = 1;
	img[0][1] = 2;
	img[0][2] = 4;
	img[0][3] = 3;
	img[1][0] = 5;
	img[1][1] = 0;
	img[1][2] = 1;
	img[1][3] = 2;
	img[2][0] = 1;
	img[2][1] = 3;
	img[2][2] = 4;
	img[2][3] = 2;
	img[3][0] = 6;
	img[3][1] = 8;
	img[3][2] = 3;
	img[3][3] = 7;

	ker[0][0] = 1;
	ker[0][1] = 2;
	ker[1][0] = 0;
	ker[1][1] = 1;

	double** c = convolute(img, 4, 4, ker, 2, 2, 1);
	print(c, 3, 3);

	for (int i = 0; i < 4; i++)
		free(img[i]);
	free(img);
	for (int i = 0; i < 2; i++)
		free(ker[i]);
	free(ker);
	for (int i = 0; i < 3; i++)
		free(c[i]);
	free(c);*/

	/*
	double** ximg = new double*;
	ximg[0] = new double[25];
	for (int i = 0; i < 25; i++)
		ximg[0][i] = i + 1;
	double** xcore = new double*;
	xcore[0] = new double[4];
	for (int i = 0; i < 4; i++)
		xcore[0][i] = 1;
	Matrix mimg;
	mimg.h = 5;
	mimg.w = 5;
	mimg.matrix = ximg;
	Matrix mcore;
	mcore.h = 2;
	mcore.w = 2;
	mcore.matrix = xcore;
	link(mimg, mcore, 2);
	*/

	
	
	/*Matrix ret;
	ret.h = 4;
	ret.w = 4;
	ret.matrix = (double**)malloc(ret.h * sizeof(double*));
	for (int i = 0; i < 4; i++) {
		ret.matrix[i] = (double*)malloc(4 * sizeof(double));
		for (int j = 0; j < 4; j++)
			ret.matrix[i][j] = i * 4 + j + 1;
	}
	InitNN(3, 9, 4, 2, 1);
	print(predictConv(ret));*/

	const int input_size = 784;
	const int num_features = 10;

	FILE* lbl = fopen("D:\\ \\c++\\Convolution\\Convolution\\mnist\\mnist_labels.txt", "r");
	double*** labels = (double***)malloc(train_size * sizeof(double**));
	Matrix* y_train = (Matrix*)malloc(train_size * sizeof(Matrix));
	int d = 0;
	//binary multiclassification:
	for (int i = 0; i < train_size; i++) {
		labels[i] = (double**)malloc(num_features * sizeof(double*));
		fscanf(lbl, "%d", &d);
		for (int j = 0; j < num_features; j++) {
			labels[i][j] = (double*)malloc(sizeof(double));
			labels[i][j][0] = 0.0;
		}
		labels[i][d][0] = 1.0;
		y_train[i].h = num_features;
		y_train[i].w = 1;
		y_train[i].matrix = labels[i];
	}
	fclose(lbl);

	Matrix* x_train = new Matrix[train_size];
	FILE* pic;
	for (int i = 0; i < train_size; i++) {
		x_train[i].matrix = (double**)malloc(input_size * sizeof(double*));
		x_train[i].h = input_size;
		x_train[i].w = 1;
		char* path = new char[16];
		path[0] = 0;
		char* num = new char[10];
		num = _itoa(i, num, 10);
		strcat(num, ".txt");
		strcat(path, "mnist\\");
		strcat(path, num);
		pic = fopen(path, "r");
		if (pic == NULL) {
			printf("File does not exist: %d.txt\n", i);
			continue;
		}
		for (int j = 0; j < input_size; j++) {
			x_train[i].matrix[j] = (double*)malloc(sizeof(double));
			fscanf(pic, "%f", &d);
			x_train[i].matrix[j][0] = d;
		}
		free(path);
		free(num);
		fclose(pic);
	}

	//Initialize neural network with 3 layers:
	InitNN(3, 729, 196, 49, 10);
	//double* mse = (double*)malloc(iterations * sizeof(double));//mean-squared error
	FILE* mse = fopen("D:\\ \\c++\\Convolution\\Convolution\\mse.txt", "w");
	//training:
	//print_weights();
	

	for (int i = 0; i < epochs * train_size; i++) {
		fprintf(mse, "%f\n", backpropagation(x_train[i % (train_size - 10)], y_train[i % (train_size - 10)], 0.515));
	}
	printf("Done!\n");
	for (int i = 0; i < 10; i++) {
		int n = train_size - i - 1;
		for (int j = 0; j < 10; j++) {
			if (y_train[n].matrix[j][0])
				printf("Input: %d\n\n\n", j);
		}
		Matrix prediction = predict(x_train[n]);
		double max = 0.0;
		int p = 0;
		for (int k = 0; k < 10; k++) {
			if (prediction.matrix[k][0] > max) {
				max = prediction.matrix[k][0];
				p = k;
			}
		}
		printf("Prediction: %d\n\n\n", p);
	}
	
	getchar();
	getchar();
}