//============================================================================
// Name        : GPmix.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "gpmix/gp/gp_base.h"
//#include "gpmix/matrix/matrix_helper.h"
using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	unsigned int n=10;
	unsigned int m=5;
	MatrixXd X;
	X = randn( n, m);
	return 0;
}
