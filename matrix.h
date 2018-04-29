/* 
 * File:   matrix.h
 * Author: michele
 *
 * Created on 12 February 2018, 15:53
 */

#ifndef MATRIX_H
#define	MATRIX_H

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <string.h>
//#include <malloc.h>
#include <iostream>
#include <cstdlib>
//#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <ctime>

//threads block size for kernel launch
#define blockSize 1024

//macro for CUDA kernels error handling
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

 inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//macro for CURAND error handling 
#define CURAND_CALL(x) { curandAssert((x), __FILE__, __LINE__); }

inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CURAND_STATUS_SUCCESS) 
   {
      fprintf(stderr,"CURAND error at: %s %d\n %d\n", file, line, curandStatus(code));
      if (abort) exit(code);
   }
}
/*
 * 
 */
////////////////////////////
///////HOST MATRIX//////////
////////////////////////////
//matrix class allocated in host memory 

class matrix{
public:
    int X;
    int Y;
    double * V;
    //class constructor
    matrix(){};
    matrix(int X,int Y,double val);
    //destructor
    ~matrix(){delete [] V;};
    //parentheses operator,  A(i,j) = A_i,j element of matrix A
__host__ __device__    double & operator()(const int& i, const int& j);
    //print to screen matrix
    void display();
    //save matrix to file in ascii format
    void save(std::string filename);
    //loads matrix from ascii
    void load(std::string filename);
};


//class constructor initialise matrix entries to val
matrix::matrix(int X, int Y, double val = 0){
            this->X=X;
            this->Y=Y;
            this->V=new double [X*Y];
            for(int i=0; i<X*Y; ++i){
                V[i]=val;}
   }
//parentheses operator,  A(i,j) = A_i,j element of matrix A
double & matrix::operator()(const int& i ,const int& j){
    return V[j+Y*i];
}

//print to screen matrix
void matrix::display(){
    for(int i = 0; i < X; ++i){
        for (int j = 0; j< Y; ++j){
            std::cout<<V[j+i*Y]<<" ";
        }
        std::cout<<"\n";
    }
}

/////////////////////
//////LOAD DATA//////
/////////////////////
//loading data from MNIST database http://yann.lecun.com/exdb/mnist/, code taken from
//https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set///

//reverse integer written in binary form
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

//reads image values from MNIST database, 
//stores them in a preexisting matrix
void ReadMNIST_Images(int NumberOfImages, int DataOfAnImage, matrix& arr, std::string filename, bool verbose, int offset = 0){
    std::ifstream file;
    file.open(filename.c_str(),std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= NumberOfImages;//ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        
//        unsigned char buffer[offset*n_rows*n_cols];
//        file.read((char*)&buffer,sizeof(buffer));  
        file.ignore(offset*n_rows*n_cols*sizeof(unsigned char));
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    //std::cout<<" weee! "<<i<<"\t"<<r<<"\t"<<c<<std::endl;
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr(i,(n_rows*r)+c)= (double)temp/255.0;
                }
            }
        }
        if(verbose)
            std::cout<<"------------------\nimage data loaded\n------------------\n";
        file.close();
    }
    else{
        std::cout<<"could not open "<<filename<<std::endl;
        exit(1);
    }
}
//reads labels from MNIST database
//stores them in existing array
void ReadMNIST_Labels(int NumberOfLabels, matrix& arr, std::string filename,bool verbose, int offset = 0){
    std::ifstream file;
    file.open(filename.c_str(),std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_items=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items= NumberOfLabels; //ReverseInt(number_of_items);
        
        
//        unsigned char buffer[offset];
//        file.read((char*)&buffer,sizeof(buffer));
//        
        file.ignore(offset*sizeof(unsigned char));
        for(int i = 0; i < number_of_items; ++i){
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            arr(i,0)= (double)temp;
        }
        file.close();
        if(verbose)
            std::cout<<"------------------\nlabel data loaded\n------------------\n";
    }
    else{
        std::cout<<"could not open "<<filename<<std::endl;
        exit(1);
    }
    
}
//read matrix values from ascii file
void matrix::load(std::string filename){
    std::ifstream input;
    input.open(filename.c_str()); 
    if(input.is_open()){
        for(int i=0; i<X;++i){
            for(int j=0; j<Y;++j){
                input>>this->operator ()(i,j);
            }
        }
        input.close();
    }
    else{
        std::cout<<"could not open "<<filename<<std::endl;
        exit(1);
    }
}
//save matrix values to ascii file
void matrix::save(std::string filename){
    std::ofstream output;
    output.open(filename.c_str());   
    for(int i=0; i<X;++i){
        for(int j=0; j<Y;++j){
           output<<this->operator ()(i,j)<<std::endl;
        }
    }
    output.close();
}

//////////////////////////////////////////////
//DEVICE MATRIX///////////////////////////////
//////////////////////////////////////////////
//matrix class in GPU device

class d_matrix{
public:
    int X;
    int Y;
    double * V = NULL;
    bool is_cpy;
    //needed for assignment = operator
    friend void swap(d_matrix& first, d_matrix& second){
        using std::swap;
        swap(first.X,second.X);
        swap(first.Y,second.Y);
        swap(first.V,second.V);  
        swap(first.is_cpy,second.is_cpy);
    }
    d_matrix(){};
    //constructor
    d_matrix(int X,int Y, double val);
    //copy-constructor
    d_matrix(const d_matrix& rhs);
    //destructor
    ~d_matrix(){
       if(!is_cpy){
            if (V!=NULL){
                gpuErrchk(cudaFree(V));
                gpuErrchk(cudaPeekAtLastError());
                V=NULL;
            }
       }
    };
    
    
    //parentheses operator: A(i,j) returns element A_i,j of matrix A    
__host__ __device__    double & operator()(const int& i,const int& j);


    d_matrix operator()(const int& i0, const int& iM, const int& j0, const int& jN);

    //copy device matrix content to host matrix
    void toHost(const matrix& A);
    
    //copy host matrix content to device matrix
    void toDev(const matrix& A);
    
    //product of matrices
    d_matrix operator*(const d_matrix& B);
    
    //sum of matrices
    d_matrix operator+(const d_matrix& B);
    
    //multiplication by scalar
    d_matrix operator*(double c);
    
    //assignment operator
    d_matrix& operator=(d_matrix B);
    
    //returns transpose matrix
    d_matrix transpose();
    
    //element-wise multiplication of matrices (Hadamard product)
    d_matrix dot(const d_matrix& B);
    
    //from matrix A returns matrix of size (batch X columns)
    //extracting #batch rows from A starting from row i
    d_matrix get_row(const int i, int batch);
    
    //add a row of val at position row and returns new matrix of size (m+1 x n)
    d_matrix add_row(int val);
    
    //one-hot encoder operator, returns boolean sparse matrix B
    //where B(i,j) == 1 iff A(i)==j
    d_matrix dummify(int max);
    
    //sum of all elements of the matrix
    double sum();
    
    //sum along rows
    d_matrix sum_rows();
    
    //max element of a matrix
    double max();
    
    //display matrix to screen
    void display();
    //display matrix dimensions to screen
    void display_dim();
    //random initialisation of matrix
    void randomize(double mean, double stdDev);
    
    //rearrange rows ordering of a matrix according to order vector
    void arrange_rows(int * order);
    
    //load matrix from ascii
    void load(std::string filename);
    
    //save matrix to ascii
    void save(std::string filename);
};


//kernel function sets the value of all elements of A to c
__global__ void setValueKernel(double * A, double c, int N){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx< N){ 
        A[idx]=c;
    }
}

//class constructor
d_matrix::d_matrix(int X,int Y, double val=0){
            this->X=X;
            this->Y=Y;
            gpuErrchk( cudaMalloc((void**)&this->V, X*Y*sizeof(double)) ); 
            gpuErrchk( cudaMemset(this->V,0, X*Y*sizeof(double)) );
            int gridSize = ceil(X*Y/(float)blockSize);
            setValueKernel<<<gridSize,blockSize>>>(this->V,val,this->X*this->Y);
            gpuErrchk( cudaDeviceSynchronize() );
            gpuErrchk( cudaPeekAtLastError() ); 
            this->is_cpy=false;}

//class copy-constructor
d_matrix::d_matrix(const d_matrix& rhs): V(rhs.V),X(rhs.X),Y(rhs.Y){ 
    this->is_cpy=true;
    gpuErrchk( cudaMemcpy (V, rhs.V, X*Y*sizeof(double), cudaMemcpyDeviceToDevice) );
    
}


//bracket operator A(i,j) = A_i,j
double & d_matrix::operator ()(const int& i, const int& j){
    return this->V[j+this->Y*i];
}


/// copy submatrix of A into B
__global__ void cropMatrixKernel(double * A, d_matrix B, int i0, int j0, int Acols){
    int idx = threadIdx.x+ blockIdx.x*blockDim.x;
    if(idx < B.X*B.Y){ 
        int i = idx/B.Y;
        int j = idx%B.Y;
        B(i,j) = A[(i+i0)*Acols+j+j0];
    }
}


//select range of consecutive rows and columns and create new matrix
d_matrix d_matrix::operator()(const int& i0, const int& iM, const int& j0, const int& jN){
    d_matrix B(iM-i0,jN-j0);    
    int gridSize = ceil(B.X*B.Y/(float)blockSize);
    cropMatrixKernel<<<gridSize,blockSize>>>(this->V,B,i0,j0,this->Y);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );     
    return B;    
}


//copy device matrix content to host matrix
void d_matrix::toHost(const matrix& A){
    if(A.X==X && A.Y==Y )
    cudaMemcpy (A.V, V, A.X*A.Y*sizeof(double), cudaMemcpyDeviceToHost);
    else{
        std::cout<<"invalid dimensions\n";
        exit(1);
    }
}

//copy host matrix content to device matrix
void d_matrix::toDev(const matrix& A){
    if(A.X==X && A.Y==Y )
    cudaMemcpy (V, A.V, A.X*A.Y*sizeof(double), cudaMemcpyHostToDevice);
    else{
        std::cout<<"invalid dimensions\n";
        exit(1);
    }
}


//kernel for multiplication of matrices
__global__ void multiplicationKernel(double * A, d_matrix B, d_matrix C, int m){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx< m*B.Y){
        int i = idx/B.Y;
        int j = idx%B.Y;
        double temp=0;
        for(int k = 0 ; k< B.X; ++k){
            temp += A[i*B.X+k]*B(k,j);
        }
        C(i,j)=temp;
        
    }
    
}


//matrix multiplication operator
d_matrix d_matrix::operator*(const d_matrix& B){
    d_matrix C(this->X,B.Y);
    if(this->Y == B.X){        
        int gridSize = ceil(this->X*B.Y/(float)blockSize);
        multiplicationKernel<<<gridSize,blockSize>>>(this->V,B,C,this->X);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() ); 
    }
    else{
        std::cout<<"Matrices dimensions not compatible for multiplication\n"<<
                    "("<<this->X<<"x"<<this->Y<<") vs ("<<B.X<<"x"<<B.Y<<")\n";
        exit(1);
    }
    return C;
}

//kernel for matrix sum
__global__ void sumKernel(double * A, double * B, double * C, int N){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < N ) C[idx]=A[idx]+B[idx];

}


//kernel for sum of matrix with column vector (column-wise sum)
__global__ void sumKernelVec(double * A, double * B, double * C, int X, int Y){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < X*Y ) {
        int row = idx/Y;
        C[idx]=A[idx]+B[row];
    }

}

//sum of matrices of same size or sum of matrix with column vector
d_matrix d_matrix::operator+(const d_matrix& B){
    d_matrix C(this->X,this->Y);
    if(this->Y == B.Y && this->X == B.X){        
        int gridSize = ceil(this->X*B.Y/(float)blockSize);
        sumKernel<<<gridSize,blockSize>>>(this->V,B.V,C.V,this->X*this->Y);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() ); 
    }
    else if((this->X == B.X && B.Y == 1 )){
        int gridSize = ceil(this->X*B.Y/(float)blockSize);
        sumKernelVec<<<gridSize,blockSize>>>(this->V,B.V,C.V,this->X,this->Y);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() ); 
    }
    else{
        std::cout<<"Matrices dimensions not compatible for addition\n"<<
                    "("<<this->X<<"x"<<this->Y<<") vs ("<<B.X<<"x"<<B.Y<<")\n";
        exit(1);
    }
    return C;
}


//kernel for scalar multiplication
__global__ void scalarMultiplicationKernel(double * A,double * B, double c,int N){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < N ) B[idx]=A[idx]*c;
}

//scalar multiplication operator
d_matrix d_matrix::operator*(double c){
    d_matrix C(this->X,this->Y);        
    int gridSize = ceil(this->X*this->Y/(float)blockSize);
    scalarMultiplicationKernel<<<gridSize,blockSize>>>(this->V,C.V,c,this->X*this->Y);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() ); 
    return C;
}


//assignment operator
d_matrix& d_matrix::operator=(d_matrix B){
    swap(*this,B);
    return *this;
}


//kernel for transpose operation
__global__ void transpositionKernel(double * A, d_matrix B){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx< B.X*B.Y){
        int i = idx/B.X;
        int j = idx%B.X;
        B(j,i)=A[idx];
        
    }
}


//kernel for element-wise multiplication of matrices
__global__ void pointwiseMultiplicationKernel(double * A, double * B, double * C, int N){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx< N){
        C[idx]=A[idx]*B[idx];
    }
    
}


//element-wise multiplication of matrices (Hadamard product)
d_matrix d_matrix::dot(const d_matrix& B){
    d_matrix C(this->X,this->Y);
    if(this->X == B.X && this->Y == B.Y){
        
        int gridSize = ceil(this->X*this->Y/(float)blockSize);
        pointwiseMultiplicationKernel<<<gridSize,blockSize>>>(this->V,B.V,C.V,this->X*this->Y);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() ); 
    }
    else{
        std::cout<<"Matrices dimensions not compatible for Hadamard product\n"<<
                    "("<<this->X<<"x"<<this->Y<<") vs ("<<B.X<<"x"<<B.Y<<")\n";
        exit(1);
    }
    return C;
}


//transposition of matrix
d_matrix d_matrix::transpose(){
    d_matrix temp(this->Y,this->X);
    int gridSize = ceil(this->X*this->Y/(float)blockSize);
    transpositionKernel<<<gridSize,blockSize>>>(this->V,temp);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() ); 
//    swap(*this,temp);
//    return *this;
    return temp;
}


//displays matrix to screen
void d_matrix::display(){
    matrix temp(this->X,this->Y);
    this->toHost(temp);
    temp.display();
}

void d_matrix::display_dim(){
    std::cout<<"("<<this->X<<"X"<<this->Y<<")"<<std::endl;
}

//load matrix from ascii
void d_matrix::load(std::string filename){
    matrix temp(this->X,this->Y);
    temp.load(filename);
    this->toDev(temp);
    
}

//save matrix to ascii
void d_matrix::save(std::string filename){
    matrix temp(this->X,this->Y);
    this->toHost(temp);
    temp.save(filename);
}


//set matrix elements to norm. dist. values
void d_matrix::randomize(double mean, double stdDev){
    curandGenerator_t generator;
    CURAND_CALL( curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(generator, time(NULL)) );

    //gpuErrchk(cudaMalloc((void**)&rand, Geo.randDim*sizeof(float)));
    int rand_dim = ((this->X*this->Y&1)==0 ? this->X*this->Y : this->X*this->Y+1); 
    CURAND_CALL( curandGenerateNormalDouble(generator, this->V, rand_dim, mean, stdDev) );
    CURAND_CALL( curandDestroyGenerator(generator) );
}


//kernel for parallel sum of elements
__global__ void block_sum(double* input, double*  per_block_results, const size_t n) {
    extern __shared__ double sdataD[];

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into __shared__ memory
    double x = 0.0;
    if (tid < n) {
        x = input[tid];
    }
    sdataD[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            // add a partial sum upstream to our own
            sdataD[threadIdx.x] += sdataD[threadIdx.x + offset];
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0) {
        per_block_results[blockIdx.x] = sdataD[0];
    }
}


//sum of all elements of matrix
double d_matrix::sum(){
    double total_sum=0;
    int gridSize = ceil(this->X*this->Y/(float)blockSize);
    int num_blocks=gridSize;
    int num_elements=X*Y;
    double *d_partial_sums_and_total;
    double * temp=V; //otherwise when I do input=d_partial_sum i am losing the pointer to the original field array!!!   
    int bytes=sizeof(double)*blockSize;
   /*allocate mem in the device*/
  gpuErrchk(cudaMalloc((void **)&d_partial_sums_and_total, (num_blocks)*sizeof(double)));
  
  /*initialize allocated memory in the device*/
  
  gpuErrchk(  cudaMemset(d_partial_sums_and_total, 0, (num_blocks)*sizeof(double)) );

// launch one kernel to compute, per-block, a partial sum
  while(num_blocks>1){
  block_sum<<<num_blocks,blockSize,bytes>>>(temp, d_partial_sums_and_total, num_elements);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  num_elements=num_blocks;
  num_blocks=num_blocks/blockSize +1;
  temp=d_partial_sums_and_total; 
  
  }
  block_sum<<<num_blocks,blockSize,bytes>>>(temp, d_partial_sums_and_total, num_elements);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  /*copy to host to complete the sum serial*/
  gpuErrchk( cudaMemcpy(&total_sum, d_partial_sums_and_total, sizeof(double), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaFree(d_partial_sums_and_total) );
  return total_sum;  

}

//kernel for sum along rows
__global__ void dim_sumKernel(double * R, double * A, int X, int Y){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < X  ){
    double temp = 0;
 //idx represents the row in which we are
    for( int i = 0; i < Y; i++){
        temp = temp + A[Y*idx+i];
        }
    R[idx] = temp;
    }
}


//returns column vector, sum of matrix columns
d_matrix d_matrix::sum_rows( ){
        d_matrix R(this->X,1);
        int gridSize = ceil(this->X/(float)blockSize);
        dim_sumKernel<<<gridSize,blockSize>>>(R.V, this->V, this->X,this->Y);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        return R;
}


//extract #batch consecutive rows from matrix starting from row i
__global__ void getRowKernel(double * A, double * R, int i, int batch, int Y){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < batch*Y){
        R[idx]=A[idx+Y*i];
    }
}


//from matrix A returns matrix of size (columns of A X Batch)
//extracting #batch rows from A starting from row i
//
d_matrix d_matrix::get_row(const int i, int batch = 1){
    d_matrix row(batch,this->Y);                        // row sizes = (batch X A_cols)
    int gridSize = ceil((batch*this->Y)/(float)blockSize);
    getRowKernel<<<gridSize,blockSize>>>(this->V,row.V,i,batch,this->Y);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
    return row.transpose();
    
}


//copy self matrix A into larger matrix B starting from position i0,j0
__global__ void copyMatrixKernel(double * A, d_matrix B, int i0, int j0,int Arows, int Acols){
    int idx = threadIdx.x+ blockIdx.x*blockDim.x;
    if(idx < Arows*Acols){ 
        int i = idx/Arows;
        int j = idx%Acols;
        B(i+i0,j+j0) = A[idx];
    }
}

d_matrix d_matrix::add_row(int val){
    d_matrix B(this->X+1,this->Y,val);
    int gridSize = ceil((this->X*this->Y)/(float)blockSize);
    copyMatrixKernel<<<gridSize,blockSize>>>(this->V,B,1,0,this->X,this->Y);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
    return B;
}



//kernel for one-hot-encoding
__global__ void dummifyKernel(double * Y0, d_matrix Y, int N){
    int idx = threadIdx.x+ blockIdx.x*blockDim.x;
    if (idx < N){
        int row = idx /Y.Y;
        Y(row,int(Y0[row])) = 1 ;
    }
}


//one-hot encoder operator, returns boolean sparse matrix B
//where B(i,j) == 1 iff A(i)==j
d_matrix d_matrix::dummify(int max){
    
    d_matrix dummy(this->X,max);
    int gridSize = ceil(this->X*max/(float)blockSize);
    dummifyKernel<<<gridSize,blockSize>>>(this->V,dummy,this->X*max);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
    return dummy;
}


//kernel for parallel computation of max element
__global__ void block_max(double* input, double*  per_block_results, const size_t n) {
    extern __shared__ double sdataM[];

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into __shared__ memory
    double x = 0.0;
    if (tid < n) {
        x = input[tid];
    }
    sdataM[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            // add a partial sum upstream to our own
            sdataM[threadIdx.x] = max(sdataM[threadIdx.x],sdataM[threadIdx.x + offset]);
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0) {
        per_block_results[blockIdx.x] = sdataM[0];
    }
}

//returns the maximum value of matrix elements
double d_matrix::max(){
    double total_sum=0;
    int num_blocks=ceil(this->X*this->Y/(float)blockSize);
    int num_elements=X*Y;
    double *d_partial_sums_and_total;
    double * temp=V; //otherwise when I do input=d_partial_sum i am losing the pointer to the original field array!!!   
    int bytes=sizeof(double)*blockSize;
    /*allocate mem in the device*/
    gpuErrchk(cudaMalloc((void **)&d_partial_sums_and_total, (num_blocks)*sizeof(double)));
    /*initialize allocated memory in the device*/
    gpuErrchk(  cudaMemset(d_partial_sums_and_total, 0, (num_blocks)*sizeof(double)) );
    // launch one kernel to compute, per-block, a partial sum
    while(num_blocks>1){
        block_max<<<num_blocks,blockSize,bytes>>>(temp, d_partial_sums_and_total, num_elements);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        num_elements=num_blocks;
        num_blocks=num_blocks/blockSize +1;
        temp=d_partial_sums_and_total; 
  }
  block_max<<<num_blocks,blockSize,bytes>>>(temp, d_partial_sums_and_total, num_elements);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());  
  gpuErrchk( cudaMemcpy(&total_sum, d_partial_sums_and_total, sizeof(double), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaFree(d_partial_sums_and_total) );
  return total_sum; 
}

//kernel for matrix rows rearrangement
__global__ void arrangeRowsKernel(double*  V, d_matrix A, int * order){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < A.X*A.Y){
        int i = idx/A.Y;
        int j = idx%A.Y;
        A(i,j) = V[A.Y*order[i]+j];
    }
}

//rearrange rows of matrix according to order vector
void d_matrix::arrange_rows( int * order){
    d_matrix Anew(this->X,this->Y);
    int gridSize = ceil((this->X*this->Y)/(float)blockSize);
    int * d_order;
    gpuErrchk( cudaMalloc((void**)&d_order, this->X*sizeof(int)) ); 
    gpuErrchk( cudaMemset(d_order,0, this->X*sizeof(int)) );
    gpuErrchk(cudaMemcpy (d_order, order, this->X*sizeof(int), cudaMemcpyHostToDevice) );
    arrangeRowsKernel<<<gridSize,blockSize>>>(this->V,Anew,d_order);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError()); 
    gpuErrchk(cudaMemcpy (this->V, Anew.V, this->X*this->Y*sizeof(double), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaFree(d_order) ) ;
    gpuErrchk(cudaPeekAtLastError()); 
}


///////////////////////////////////
///NON-MEMBER FUNCTIONS////////////
///////////////////////////////////


//non-member matrix multiplication
void multi(d_matrix& A, d_matrix& B, d_matrix& C){
    if(A.Y == B.X){
        int gridSize = ceil(A.X*B.Y/(float)blockSize);
        multiplicationKernel<<<gridSize,blockSize>>>(A.V,B,C,A.X);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaPeekAtLastError() ); 
    }
    else{
        std::cout<<"Matrices dimensions not compatible for multiplication\n"<<
                    "("<<A.X<<"x"<<A.Y<<") vs ("<<B.X<<"x"<<B.Y<<")\n";
        exit(1);
    }
}


//kernel that applies sigmoid function at every element of a matrix
__global__ void sigmoidKernel(double * A, double * B, int N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N){
        B[idx] = 1./(1.+exp(-A[idx]));
    }
}

//apply sigmoid function at each element of a matrix
d_matrix sigmoid(const d_matrix& A){
    d_matrix B(A.X,A.Y);
    int gridSize = ceil(A.X*A.Y/(float)blockSize);
    sigmoidKernel<<<gridSize,blockSize>>>(A.V,B.V,A.X*A.Y);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() ); 
    return B;
    
}


//sigmoid derivative kernel sigma'=sigma(1-sigma)
__global__ void sigmoidPrimeKernel(double * A, double * B, int N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N){
        double sig = 1./(1.+exp(-A[idx]));
        B[idx] = sig * (1.- sig);
    }
}


//apply sigma' to each matrix element
d_matrix sigmoidPrime(const d_matrix& A){
    d_matrix B(A.X,A.Y);
    int gridSize = ceil(A.X*A.Y/(float)blockSize);
    sigmoidPrimeKernel<<<gridSize,blockSize>>>(A.V,B.V,A.X*A.Y);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() ); 
    return B;
    
}


//logarithm kernel
__global__ void logKernel(double * A, double * B, int N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N){
        B[idx] = log(A[idx]);
    }
}


//apply logarithm to all elements of matrix
d_matrix logM(const d_matrix& A){
    d_matrix B(A.X,A.Y);
    int gridSize = ceil(A.X*A.Y/(float)blockSize);
    logKernel<<<gridSize,blockSize>>>(A.V,B.V,A.X*A.Y);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() ); 
    return B;
    
}


//print two dashed lines
void skip(){
std::cout<<"------------------------------\n------------------------------\n";

}

#endif	/* MATRIX_H */

