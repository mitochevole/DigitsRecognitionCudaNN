/* 
 * File:   main.cu
 * Author: michele
 *
 * Created on 12 February 2018, 12:08
 */

#include "utilities.h"



  
int main(int argc, char** argv) {

//    d_matrix A(3,4,2);
//    d_matrix W(4,3,1);
//    d_matrix B(3,1,1);
//    d_matrix C = A*W + B;
//    A.display();
//    W.display();
//    B.display();
//    C.display();
//    


    int maxSize = 10000;
    int sizeStep = 100;
    int batchSize = 10;
    int epochs = 30;
    double lambda = 5.;
    double eta = 0.001;
    bool out_cost = false;
    bool verbose = false;
    bool monitor_accuracy = false;
    bool print_accuracy = true;
    std::string outName = "/home/michele/Desktop/learning_curve.txt";
    double cost_cv=0;
    
    const int Layers = 3;
    int sizes[Layers] = {im_size,100,10};
    NeuralNetwork NN(Layers,sizes);
    NN.train(maxSize, eta, lambda, epochs, batchSize,out_cost, verbose, monitor_accuracy, print_accuracy);
    std::string label = "prova_Load-Save";
    NN.saveWeights(label);
    double test = NN.test( 5000, im_train_name, label_train_name,true,cost_cv,batchSize,false, maxSize);
    std::cout<<"test: "<<test<<" cost: "<<cost_cv<<std::endl;
    NN.loadWeights("weights_"+label);
    test = NN.test( 5000, im_train_name, label_train_name,true,cost_cv,batchSize,false, maxSize);
    std::cout<<"test: "<<test<<" cost: "<<cost_cv<<std::endl;
//    learningCurve(outName,  eta,  lambda,  batchSize,  epochs, sizeStep,  maxSize, out_cost);

    return 0;
}