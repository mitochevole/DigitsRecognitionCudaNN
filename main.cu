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
//    d_matrix B(,1,1);
//    d_matrix C = A*W + B;
//    A.display();
//    W.display();
//    B.display();
//    C.display();
//    d_matrix S(7,5);
//    S.randomize(0.,1.);
//    S.display();
//    d_matrix F = S(3,6,2,5);
//    F.display();
//    F.add_row(1.0).display();
//    F.add_row(1.0).transpose().display();
//    


    int maxSize = 40000;
//    int sizeStep = 100;
    int batchSize = 10;
    int epochs = 30;
    double lambda = 2.0;
    double eta = 0.5; //if Lambda = 0, eta = 3
    bool out_cost = true;
    bool verbose = false;
    bool monitor_accuracy = true;
    bool print_accuracy = false;
    std::string outName = "/home/michele/Desktop/learning_curve.txt";
    double cost_cv=0;
    
    const int Layers = 3;
    int sizes[Layers] = {im_size,30,10};
    NeuralNetwork NN(Layers,sizes);
    NN.train(maxSize, eta, lambda, epochs, batchSize,out_cost, verbose, monitor_accuracy, print_accuracy);
    std::string label = "prova_Load-Save";
    NN.saveWeights(label);
    double test = NN.test( 5000, im_test_name, label_test_name,true,cost_cv,batchSize,false, 0);
    std::cout<<"test: "<<test<<" cost: "<<cost_cv<<std::endl;
    
    NeuralNetwork NN2(Layers,sizes);
    NN2.loadWeights("weights_"+label);
    test = NN2.test( 5000, im_train_name, label_train_name,true,cost_cv,batchSize,false, maxSize);
    std::cout<<"test: "<<test<<" cost: "<<cost_cv<<std::endl;
//  //  learningCurve(Layers, sizes, outName,  eta,  lambda,  batchSize,  epochs, sizeStep,  maxSize, out_cost);

    return 0;
}