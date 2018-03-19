#include "mpi.h"
#include <iostream>
#include <ctime>

using namespace std;

char** allocMatrix(int rows, int cols)
{
	char **mat = new char*[rows];
	char *t = new char[rows * cols];
	for (int i = 0; i < rows; ++i)
		mat[i] = &t[i * cols];
	return mat;
}

void printMatrix(char **mat, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << (int)mat[i][j] << "  ";
		}
		cout << endl;
	}
}

int getBlockSize(int procId, int procN, int sizeRow, int sizeCols)
{
	int result = ((procId + 1)*(sizeRow) / (procN)-(procId)*(sizeRow) / (procN))*sizeCols;
	return result;
}

int getBlockDispls(int procId, int procN, int sizeRow, int sizeCols)
{
	int result = ((procId)*(sizeRow) / (procN))*sizeCols;
	return result;
}

//�������� ��������
char** linear(char **mat, int rows, int cols, char newLow, char newHigh)
{
	char** result;
	int linMin = 255, linMax = 0;
	
	//��������������� ������� ������� � �������� �������� �����������
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (mat[i][j] < linMin)
				linMin = mat[i][j];
			if (mat[i][j] > linMax)
				linMax = mat[i][j];
		}
	}
	
	//��������� ������ ��� ������� ���������� ������ ��������� ���������
	result = new char*[rows];
	char *t = new char[rows * cols];
	for (int i = 0; i < rows; ++i)
		result[i] = &t[i * cols];

	//���������������� ���������� ����� �������� ��������
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result[i][j] = ((mat[i][j] - linMin)*((newHigh - newLow) / (float)(linMax - linMin))) + newLow;
		}
	}
	return result;
}

int main(int argc, char* argv[])
{
	srand(time(0));

	int procNum, procRank;
	int rowsNum = 0, colsNum = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	
	//���������� ��� ������� ������� ������
	double tL, tP;
	
	//���������� ��� �������� � ����� ��������� � ����������
	int localXL = 255, localXH = 0, globalXL = 255, globalXH = 0, newLowBound = 0, newHighBound = 0;
	
	//�������� �������
	char **matrix;
	char **linearResultMat;

	if (procRank == 0)
	{
		cout << endl << "Enter number of rows: " << endl;
		cin >> rowsNum;
		cout << "Enter number of cols: " << endl;
		cin >> colsNum;
		cout << "Enter new lower bound: " << endl;
		cin >> newLowBound;
		cout << "Enter new higher bound: " << endl;
		cin >> newHighBound;

		matrix = allocMatrix(rowsNum, colsNum);

		//���������� "��������" ���������� �� 1 �� 255
		cout << endl << "STARTING FILL" << endl;
		for (int i = 0; i < rowsNum; i++)
		{
			for (int j = 0; j < colsNum; j++)
				matrix[i][j] = rand() % 255 + 1;
		}
		cout << "DONE WITH FILL" << endl << endl;
		
		//���� ������� ��������� ���������������� ��������
		tL = MPI_Wtime();
		cout << "STARTING LINEAR" << endl;
		linearResultMat = linear(matrix, rowsNum, colsNum, newLowBound, newHighBound);
		cout << "LINEAR:" << MPI_Wtime() - tL << endl;
		tP = MPI_Wtime();
	}

	//�������� �������� ������� � ����� ������ �������� �������� ���� ���������
	MPI_Bcast(&rowsNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colsNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&newLowBound, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&newHighBound, 1, MPI_INT, 0, MPI_COMM_WORLD);

	char **resultMat = allocMatrix(rowsNum, colsNum);
	
	//���������� ���������� ��������� ������� ����� ����������
	int *sendcounts = new int[rowsNum];
	int *displacements = new int[rowsNum];	
	int remaining = (rowsNum*colsNum) % procNum;
	int sum = 0;
	for (int i = 0; i < procNum; i++)
	{
		sendcounts[i] = getBlockSize(i, procNum, rowsNum, colsNum);
		displacements[i] = getBlockDispls(i, procNum, rowsNum, colsNum);
	}
	int rows = sendcounts[procRank] / colsNum;

	//����������� ������� Scatterv ������� ������� "�������" ����� ����������
	if (procRank == 0)
	{
		MPI_Scatterv(&matrix[0][0], sendcounts, displacements, MPI_UNSIGNED_CHAR, MPI_IN_PLACE, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	}
	else if (procRank != 0)
	{
		matrix = allocMatrix(rows, colsNum);
		MPI_Scatterv(NULL, sendcounts, displacements, MPI_UNSIGNED_CHAR, &matrix[0][0], sendcounts[procRank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	}

	//������������ ����� ��������� � ���������� (������ ������� �������� �� ����� ������)
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < colsNum; j++)
		{
			if (matrix[i][j] < localXL)
				localXL = matrix[i][j];
			if (matrix[i][j] > localXH)
				localXH = matrix[i][j];
		}
	}

	//���� ����������� ��������������� ������ min � max ������� �������� � ����������� �������� �������� � ���������
	MPI_Reduce(&localXL, &globalXL, 1, MPI_UNSIGNED_CHAR, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&localXH, &globalXH, 1, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);

	MPI_Bcast(&globalXL, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(&globalXH, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	//������������ ���������� ����� �������� �������� (������ ������� �������� �� ����� ������)
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < colsNum; j++)
		{
			matrix[i][j] = ((matrix[i][j] - globalXL)*((newHighBound - newLowBound) / (float)(globalXH - globalXL))) + newLowBound;
		}
	}
	
	//���� ����������� ������ ��������� � ������ ������� �����������
	MPI_Gatherv(&matrix[0][0], sendcounts[procRank], MPI_UNSIGNED_CHAR, &resultMat[0][0], sendcounts, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	if (procRank == 0)
	{
		cout << endl << "GLOBAL MIN: " << globalXL << " GLOBAL MAX: " << globalXH << endl;
		cout << "PARALLEL:" << MPI_Wtime() - tP << endl;

		//�������� ���������� ����������� ����������������� � ������������� ����������
		bool correct = true;

		for (int i = 0; i < rowsNum; i++)
		{
			for (int j = 0; j < colsNum; j++)
			{
				if (resultMat[i][j] != linearResultMat[i][j])
					correct = false;
			}
		}

		if (correct)
			cout << endl << "CORRECT" << endl;
		else
			cout << endl << "WRONG" << endl;
	}

	for (int i = 0; i < colsNum; i++)
		delete[] linearResultMat[i];
	delete[] linearResultMat;

	for (int i = 0; i < colsNum; i++)
		delete[] matrix[i];
	delete[] matrix;


	MPI_Finalize();
	return 0;
}