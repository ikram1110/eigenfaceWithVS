#include <iostream>
#include <vector>

class Matrix
{
public:
	// attributes
	int rows;
	int columns;
	std::vector< std::vector<double> > array;

	// constructors
	Matrix();
	Matrix(int number_of_rows, int number_of_columns);
	Matrix(int number_of_rows, int number_of_columns, std::vector< std::vector<double> > elements);

	// methods
	void print();
	Matrix transpose();
	Matrix getRow(int number_of_row) const;
	friend Matrix operator* (const Matrix& a, const Matrix& b);
};
