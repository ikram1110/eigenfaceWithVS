// eigenface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include "eigen.h"
#include "config.h"

// read pgm file from the input file
// return the vector of pixel values
std::vector<double> read_pgm(std::ifstream& file) {
	std::vector<double> values;
	std::string line;
	getline(file, line); // skip P2 line
	getline(file, line); // skip comment line
	getline(file, line); // skip width height line
	getline(file, line); // skip max value line

	int val;
	while (file >> val)
	{
		values.push_back(val);
	}
	return values;
}

// writes pgm file
// represented by a matrix (must be 1xM) to the output file
void write_pgm(std::string file, Matrix* image)
{
	std::stringstream filename;
	filename << file;
	std::ofstream image_file(filename.str().c_str());
	image_file << "P2" << std::endl << "# Generate pgm" << std::endl << Width << " " << Height << std::endl << MaxValue << std::endl;
	for (int i = 0; i < M; ++i)
	{
		int val = image->array[0][i];
		if (val < 0)
		{
			val = 0;
		}
		image_file << val << " ";
	}
	image_file.close();
}

// reads the training data from the folder, specified in DataPath
std::vector< std::vector<double> > read_training_data()
{
	std::vector< std::vector<double> > array;

	// iteration over people
	for (int face = 0; face < Faces; ++face)
	{
		std::vector< std::vector<double> > facearray;
		// iteration over photos
		for (int sample = 0; sample < Samples; ++sample)
		{
			std::stringstream filename;
			filename << DataPath << face + 1 << "/" << sample + 1 << ".pgm";
			std::ifstream image(filename.str().c_str());

			if (image.is_open())
			{
				facearray.push_back(read_pgm(image));
				image.close();
			}
			else
			{
				std::cout << "image was not opened";
			}
		}

		// find the mean image
		std::vector<double> mean;
		for (int i = 0; i < M; ++i)
		{
			double sum = 0;
			for (int j = 0; j < Samples; ++j)
			{
				sum += facearray[j][i];
			}
			mean.push_back(sum / Samples);
		}
		array.push_back(mean);
	}
	return array;
}

// method to work with matrices

// scale the element of matrix to the new range
Matrix scale(Matrix m, double min = 0, double max = 255)
{
	// find the current minimum and maximum
	double m_min = m.array[0][0];
	double m_max = m.array[0][0];
	for (int r = 0; r < m.rows; ++r)
	{
		for (int c = 0; c < m.columns; ++c)
		{
			if (m.array[r][c] < m_min)
			{
				m_min = m.array[r][c];
			}
			if (m.array[r][c] < m_max)
			{
				m_max = m.array[r][c];
			}
		}
	}
	double old_range = m_max - m_min;
	double new_range = max - min;

	// create new matrix with scale elements
	Matrix result;
	result.columns = m.columns;
	result.rows = m.rows;
	for (int r = 0; r < m.rows; ++r)
	{
		std::vector<double> row;
		for (int c = 0; c < m.columns; ++c)
		{
			row.push_back((m.array[r][c] - m_min) * new_range / old_range + min);
		}
		result.array.push_back(row);
	}
	return result;
}

// recognizes the photo
int recognize(Matrix X, Matrix B, Matrix U, Matrix W)
{
	// subtract the main image
	for (int c = 0; c < M; ++c)
	{
		X.array[0][c] -= B.array[0][c];
		if (X.array[0][c] < 0)
		{
			X.array[0][c] = 0;
		}
	}

	// find weights
	Matrix Wx = Matrix(Eigenfaces, 1);
	for (int r = 0; r < Eigenfaces; ++r)
	{
		Wx.array[r][0] = (U.getRow(r) * X.transpose()).array[0][0];
	}

	// find the closest face from the training set
	double min_distance = 0;
	int image_number = 0;
	for (int image = 0; image < N; ++image)
	{
		double distance = 0;
		for (int eigenface = 0; eigenface < Eigenfaces; ++eigenface)
		{
			distance += fabs(W.array[eigenface][image] - Wx.array[eigenface][0]);
		}
		if (distance < min_distance || image == 0)
		{
			min_distance = distance;
			image_number = image;
		}
	}
	return image_number;
}

int main(int argc, const char* argv[])
{
	Matrix A = Matrix(N, M, read_training_data());
	Matrix B = Matrix(1, M);
	write_pgm("output/matrixA.pgm", &A);

	// find the mean image
	for (int c = 0; c < M; ++c)
	{
		double sum = 0;
		for (int r = 0; r < N; ++r)
		{
			sum += A.array[r][c];
		}
		B.array[0][c] = sum / N;
	}

	// output the mean image
	write_pgm("output/meanimage.pgm", &B);

	// subtract the mean from each image
	/*for (int r = 0; r < N; ++r)
	{
		for (int c = 0; c < M; ++c)
		{
			A.array[r][c] -= B.array[0][c];
			if (A.array[r][c] < 0)
			{
				A.array[r][c] = 0;
			}
		}
	}

	// output the normalized images
	for (int i = 0; i < N; ++i)
	{
		Matrix image = A.getRow(i);
		std::ostringstream filename;
		filename << "output/normalized/" << i << ".pgm";
		write_pgm(filename.str(), &image);
	}

	// find the covariance matrix
	Matrix S = A * A.transpose();

	// find eigenvectors of the covariance matrix
	Matrix V = eigensystem(&S).second.transpose();

	// find eigenfaces
	Matrix U = Matrix(Eigenfaces, M);
	for (int r = 0; r < Eigenfaces; ++r)
	{
		Matrix eigenface = V.getRow(r) * A;

		U.array[r] = eigenface.array[0];
		double norm = 0;
		for (int i = 0; i < U.columns; i++)
		{
			norm += pow(U.array[r][i], 2);
		}
		norm = sqrt(norm);
		for (int i = 0; i < U.columns; i++)
		{
			U.array[r][i] /= norm;
		}
		// output eigenface
		eigenface = scale(U.getRow(r));
		std::ostringstream filename;
		filename << "output/eigenfaces/" << r << ".pgm";
		write_pgm(filename.str(), &eigenface);
	}

	// find weights
	Matrix W = Matrix(Eigenfaces, N);
	for (int r = 0; r < Eigenfaces; ++r)
	{
		for (int c = 0; c < N; ++c)
		{
			W.array[r][c] = (U.getRow(r) * A.getRow(c).transpose()).array[0][0];
		}
	}

	// perform recognition
	if (argc == 2)
	{
		// classify the image from the arguments
		std::ifstream image(argv[1]);
		std::vector< std::vector<double> > array;
		if (image.is_open())
		{
			array.push_back(read_pgm(image));
			image.close();
		}
		else {
			std::cout << "Error: could not open image specified in the arguments";
			return 0;
		}

		Matrix X = Matrix(1, M, array);
		std::cout << recognize(X, B, U, W) + 1;
		return 0;
	}

	double accuracy = 0;
	for (int i = 1; i <= N; ++i)
	{
		// read image
		std::stringstream filename;
		filename << DataPath << i << "/" << SampleName << ".pgm";
		std::ifstream image(filename.str().c_str());
		std::vector< std::vector<double> > array;

		if (image.is_open())
		{
			array.push_back(read_pgm(image));
			image.close();
		}
		else
		{
			std::cout << "image was not opened";
		}
		Matrix X = Matrix(1, M, array);
		int image_number = recognize(X, B, U, W);

		std::cout << i << ". " << image_number + 1 << std::endl;
		if (i == image_number + 1)
		{
			accuracy = accuracy + 1;
		}
	}

	std::cout << accuracy / N; */
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
