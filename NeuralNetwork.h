/* 
 * File:   NeuralNetwork.h
 * Author: michele
 *
 * Created on 12 February 2018, 19:23
 */

#ifndef NEURALNETWORK_H
#define	NEURALNETWORK_H

#include "matrix.h"


#define im_size 784                                     //size of single image vector
#define im_train_name "mnist/train-images-idx3-ubyte"   //path to images training set
#define label_train_name "mnist/train-labels-idx1-ubyte"//path to labels training set
#define im_test_name "mnist/t10k-images-idx3-ubyte"     //path to images test set
#define label_test_name "mnist/t10k-labels-idx1-ubyte"  //path to labels test set

//neural network class
//upon initialisation the number of layers and 
//the number of nodes per layer need to be specified
//weight and bias matrices as well as z, a and delta vectors for back-propagation
//are generated upon initialisation
struct NeuralNetwork{
public:
    int L;
    int * sizes;
    d_matrix * theta = NULL;
    d_matrix * b = NULL;
    d_matrix * zeta = NULL;
    d_matrix * a = NULL;
    d_matrix * delta = NULL;
    bool is_cpy;
    //constructor
    NeuralNetwork(){};
    NeuralNetwork(int L, int * sizes){
      this->L = L;
      this->sizes = new int[L];
      this->is_cpy = false;
      for(int i = 0 ; i < L ; ++i ){
          this->sizes[i]=sizes[i];    
          
      }
      zeta = new d_matrix[L-1];
      theta = new d_matrix[L-1];
      b = new d_matrix[L-1];
      a = new d_matrix[L];
      delta = new d_matrix[L-1];
      for(int i = 0 ; i < L ; ++i ){
          a[i] = d_matrix(sizes[i],1);
          if(i<L-1){
              zeta[i] = d_matrix(sizes[i+1],1);    
//              theta[i] = d_matrix(sizes[i+1],sizes[i]+1);
              theta[i] = d_matrix(sizes[i+1],sizes[i]);
              b[i] = d_matrix(sizes[i+1],1);
              delta[i] = d_matrix(sizes[i+1],1); 
              
          }  
          
      }
      
    };
    //destructor
    ~NeuralNetwork(){
        if(!is_cpy){
            for(int l = 0; l < L; ++l){
                if (a[l].V!=NULL){
                    gpuErrchk(cudaFree(a[l].V));
                    gpuErrchk(cudaPeekAtLastError());
                    a[l].V=NULL;
                }
                if(l < L-1){
                    if (zeta[l].V!=NULL){
                    gpuErrchk(cudaFree(zeta[l].V));
                    gpuErrchk(cudaPeekAtLastError());
                    zeta[l].V=NULL;
                    }
                    if (theta[l].V!=NULL){
                    gpuErrchk(cudaFree(theta[l].V));
                    gpuErrchk(cudaPeekAtLastError());
                    theta[l].V=NULL;
                    }
                    if (b[l].V!=NULL){
                    gpuErrchk(cudaFree(b[l].V));
                    gpuErrchk(cudaPeekAtLastError());
                    b[l].V=NULL;
                    }
                    if (delta[l].V!=NULL){
                    gpuErrchk(cudaFree(delta[l].V));
                    gpuErrchk(cudaPeekAtLastError());
                    delta[l].V=NULL;
                    }
                }
            }
            delete [] sizes;
            delete [] a;
            delete [] zeta;
            delete [] theta;
            delete [] b;
            delete [] delta;
        }
    }; 
    //no copy-constructor yet :|
//    NeuralNetwork(NeuralNetwork& rhs):L(rhs.L),sizes(rhs.sizes){ // da finire
//     for(int i = 0 ; i < L ; ++i )this->sizes[i]=rhs.sizes[i];
//     this->is_cpy = true;
//    };
    
    //compute cost for single instance
    double cost(d_matrix& Y, double lambda, int batch );
    
    //feed-forward algorithm
    void feedForward();
    
    //back-propagation algorithm
    void backProp(d_matrix& Y);
    void dummyBackprop(d_matrix& X, d_matrix& Y, int batch, double eta, double lambda);
    //gradient descent
    void gradientDescent(double eta, double lambda, int batch, int n_images);
    
    //load data from MNIST 
    void load_data(int n_images, d_matrix& X, d_matrix& Y,std::string Xname, std::string Yname, bool verbose, int offset);
    
    //random initialisation of weights and biases
    void random_init();
    
    //train algorithm
    void train(int n_images_train, double eta, double lambda, int epochs, 
               int batch, bool out_cost, bool verbose, bool monitor_accuracy, bool print_accuracy);
    //compare prediction and true label
    double compare(d_matrix& Y, int batch);
    
    //test trained algorithm on the NIST training ste
    double test(int n_images_test, std::string test_data, std::string test_labels, 
                bool saveCost, double& Cost,
                int batch, bool verbose , int offset);  
    //save weights to file
    void saveWeights(std::string label);
    //load weights from file
    void loadWeights(std::string folderName);
};

//kernel to compute cross-entropy cost and avoid nan sitiations
__global__ void costKernel(d_matrix cost, d_matrix Y, d_matrix a){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < Y.X*Y.Y){
        cost.V[idx] = -(Y.V[idx]*log(a.V[idx])+(1.- Y.V[idx])*log(1-a.V[idx]) );
    }
}



//compute cost functions as cross-entropy: Sum_{i=1}^{m} y_i*log(h(x_i))+(1-y_i)*log(1-h(x_i))
double NeuralNetwork::cost(d_matrix& Y, double lambda, int batch){
    d_matrix ones(Y.X,Y.Y,1);
    d_matrix cost = (Y.dot(logM(a[L-1]))+(ones+Y*(-1.)).dot(logM(ones+a[L-1]*(-1.))))*(-1.);
    double C = cost.sum()/batch;
    for(int l = 0; l < L-1; l++){
        C += (0.5*lambda/batch)*((theta[l]).dot(theta[l])).sum();
//        C += (0.5*lambda/batch)*((theta[l](0,theta[l].X,1,theta[l].Y)).dot(theta[l](0,theta[l].X,1,theta[l].Y))*(lambda)).sum();

    }
    
    return C;
}


// compute feed forward path through the NN:
// a_0 = x_i;
// z_l = theta_l*a_l + b_l
// a_l+1 = sigma(z_l)
void NeuralNetwork::feedForward(){
    for(int l = 0; l < L-1; ++l){
        zeta[l] = (theta[l]*a[l])+b[l];
//        zeta[l] = (theta[l]*(a[l].add_row(1.0)));
        a[l+1] = sigmoid(zeta[l]);   
    }
}

//compute backpropagation algorithm
void NeuralNetwork::backProp(d_matrix& Y){
    delta[L-2] = (a[L-1] + (Y*(-1.)));
    for(int l = L-3; l >= 0; --l ){        
        delta[l] = ((theta[l+1].transpose())*delta[l+1]).dot(sigmoidPrime(zeta[l]));
//        delta[l] = ((theta[l+1].transpose())*delta[l+1])(1,theta[l+1].Y,0,delta[l+1].Y).dot(sigmoidPrime(zeta[l]));
    }    
}


//compute gradient descent
void NeuralNetwork::gradientDescent(double eta, double lambda, int batch ,int n_images){
    for(int l = 0; l < L-1; ++l){
//        theta[l] = (theta[l]*(1.-eta*lambda/n_images)) + ((delta[l]*(a[l].add_row(1.0).transpose()))*( -eta/batch));
        theta[l] = (theta[l]*(1.-eta*lambda/n_images)) + ((delta[l]*(a[l].transpose()))*( -eta/batch));
        b[l] = b[l] + ((delta[l].sum_rows())*(- eta/batch));
    }
}


//read in the MNIST images and labels from training set
void NeuralNetwork::load_data(int n_images, d_matrix& X, d_matrix& Y,std::string Xname, std::string Yname, bool verbose, int offset = 0 ){
    matrix h_X(n_images,im_size);
    ReadMNIST_Images(n_images,im_size,h_X,Xname, verbose, offset);
    
    matrix h_Y(n_images,1);
    ReadMNIST_Labels(n_images,h_Y,Yname, verbose, offset);
    X.toDev(h_X);
    Y.toDev(h_Y);
}


//random initialisation of weights and biases
void NeuralNetwork::random_init(){
    for(int l = 0; l < L-1; ++l){
        theta[l].randomize(0,1./sqrt(theta[l].Y));
        b[l].randomize(0,1.);
    }
}


//training algorithm
void NeuralNetwork::train(int n_images_train, double eta, double lambda, int epochs, int batch = 10, 
            bool out_cost=false, bool verbose = false, bool monitor_accuracy = false, bool print_accuracy = false){
    d_matrix X0(n_images_train,im_size); //X0 size is (train_set size X n_features) e.g.: 10K X 784 in case of MNIST images
    d_matrix Y0(n_images_train,1);
    this->load_data(n_images_train, X0,Y0,im_train_name,label_train_name,verbose); //read in training set images and labels
    Y0 = Y0.dummify(10);        //one-hot-encoding of labels
    d_matrix Y(10,batch);       //according to selected mini-batch size will take 
    d_matrix X(im_size,batch);  //slices of size batch from data
    
    random_init();              //initialise weights and biases
    int order[n_images_train];  //create vector of ints between 0 and size of training set-1
    for (int i = 0; i < n_images_train; i++)
        order[i]=i;    
    std::ofstream output;
    std::stringstream filename;
    if(print_accuracy){
        filename<<"accuracy_over_epochs_"<<std::time(NULL);
        output.open(filename.str());
    }
    for( int e = 0; e < epochs; ++e){
        double count = 0;
        double C = 0;
        std::srand(std::time(0));
        std::random_shuffle(&order[0],&order[n_images_train]); //shuffle order vector
        X0.arrange_rows(order);                                //use it to rearrange data at 
        Y0.arrange_rows(order);                                //beginning of each epoch       
        for(int i = 0; i < n_images_train; i+= batch){
            //the returned matrix is transposed so that 
            //features are now described along columns and each element 
            //in minibatch is a different column
            a[0] = X0.get_row(i,batch);             //assign to a_0 a minibatch of the training set. a0 =(sizes[0]X minibatch)
            Y = Y0.get_row(i,batch);                //assign to Y a minibatch of the corresponding labels Y (one-hot-encoding(10) X minibatch)
            feedForward();                          //feedforward step. obtain a prediction h(X)
            if(monitor_accuracy||print_accuracy){
                count += compare(Y,batch);
            }
            if(out_cost){
                C += cost(Y,lambda,batch);
            }
            backProp(Y);                            //use backpropagation to estimate gradients of weights theta and b            
            gradientDescent(eta,lambda,batch, n_images_train);      //perform gradient descent to improve weights
        }
        if(out_cost){                               //if true prints out the cost function for the last minibatch at the end of each epoch
            std::cout<<e<<"\t"<<(C*batch)/n_images_train<<std::endl;
            }
        if(monitor_accuracy){
            std::cout<<e<<"\t"<<count/n_images_train<<std::endl;
        }  
        if(print_accuracy)
            output<<e<<"\t"<<count/n_images_train<<std::endl;
    }
    if(output.is_open())
        output.close();
}


//finds the position of maximum in a vector in the host
void maxIndex(double * V, int Rows, int Cols, int * Idx){
    
    for (int col = 0; col < Cols; col++){
        double max=V[col];
        for (int row = 0; row < Rows; row++){
            if(V[row*Cols+col]> max){
                max = V[row*Cols+col];
                Idx[col] = row;
            }
        }
    }   
}


//kernel that compares the one-hot position of Y with the position of max in X for a batch
__global__ void compareKernel(d_matrix A, d_matrix Y, d_matrix res){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < A.Y){                             //A.Y is minibatch size and idx is the column index
        double Amax=0;
        int AmaxId = 0;
        double Ymax = 0;
        int YmaxId = 0;
        for(int j = 0; j < A.X; j++){ //A.X is 10: the length of the boolean values matrix
            if(A(j,idx) > Amax){
                Amax = A(j,idx);
                AmaxId = j;
            }
            if(Y(j,idx) > Ymax){
                Ymax = Y(j,idx);
                YmaxId = j;
            }              
        }
        res.V[idx] = int(YmaxId==AmaxId);
    }
}


//function that compares output from the activations of last layer 
//against actual labels
double NeuralNetwork::compare(d_matrix& Y, int batch = 1){
    d_matrix res(batch,1);
    int gridSize = ceil(a[L-1].Y/(float)blockSize);
    compareKernel<<<gridSize,blockSize>>>(a[L-1],Y,res);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    return (double)res.sum();
}


//test function
//load test set images and labels
//performs feedforward and compare prediction with correct label
//comparison is between the index of the maximum value returned by the hypothesis (between 0 and 9)
//and the index of the one-hot encoded label vector 
double NeuralNetwork::test(int n_images_test, std::string test_data, std::string test_labels, 
        bool saveCost, double& Cost,
        int batch = 1, bool verbose = false, int offset = 0){
    d_matrix X0(n_images_test,im_size);
    d_matrix Y0(n_images_test,1);
    this->load_data(n_images_test, X0, Y0, test_data, test_labels, verbose, offset);
    Y0 = Y0.dummify(10);    
    d_matrix Y(10,batch);
    d_matrix X(im_size,batch);
    double count = 0;
    double C = 0;
    for(int i = 0; i < n_images_test; i += batch){
        a[0] = X0.get_row(i, batch);  
        Y = Y0.get_row(i, batch); 
        feedForward();
        C += cost(Y,0,batch);  
        count +=compare(Y,batch);
    }
    if(saveCost){
        Cost = (C*batch)/(double)n_images_test;
    }
    return count/(double)n_images_test;       
}



//save weights and biases to file
void NeuralNetwork::saveWeights(std::string label){
    std::stringstream streamfile;
    streamfile<<"weights_";
    streamfile<<label.c_str();
    system(("mkdir -p "+streamfile.str()).c_str());
    for(int l = 0 ; l < L-1; l ++){
        std::stringstream filenameW,filenameB;
        filenameW<<streamfile.str()<<"/w_"<<l<<".txt";
        filenameB<<streamfile.str()<<"/b_"<<l<<".txt";
        theta[l].save(filenameW.str());
        b[l].save(filenameB.str());
    }
    
}

//load weights from file
void NeuralNetwork::loadWeights(std::string folderName){
    
    for(int l = 0 ; l < L-1; l ++){
        std::stringstream filenameW,filenameB;
        filenameW<<folderName<<"/w_"<<l<<".txt";
        filenameB<<folderName<<"/b_"<<l<<".txt";
        theta[l].load(filenameW.str());
        b[l].load(filenameB.str());
    }
    
}

//display 784 int vector image as 28x28 array with grayscale values between 0 and 255
void display_image(d_matrix& A){
    matrix h_A(A.X,A.Y);
    A.toHost(h_A);
    for(int i = 0; i < 28; i++){
        for(int j = 0; j< 28; j++){
            std::cout<<h_A.V[i*28+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

#endif	/* NEURALNETWORK_H */

