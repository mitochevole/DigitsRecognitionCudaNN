/* 
 * File:   main.cu
 * Author: michele
 *
 * Created on 12 February 2018, 12:08
 */

#include "utilities.h"



  
int main(int argc, char** argv) {


//DEFINE NN HYPERPARAMETERS

    int trainSize = atoi(argv[1]);
    int batchSize = atoi(argv[2]);
    int epochs = atoi(argv[3]);
    double lambda = atof(argv[4]);
    double eta = atof(argv[5]); //if Lambda = 0, eta = 3
    bool monitor_cost = atoi(argv[6]);
    bool print_cost =atoi(argv[7]);
    bool verbose = atoi(argv[8]);
    bool monitor_accuracy = atoi(argv[9]);
    bool print_accuracy = atoi(argv[10]);
    std::string train_label_output = argv[11];
    const int Layers = atoi(argv[12]);
    int sizes[Layers];
    for(int i = 0; i < Layers; i++){
            sizes[i]= atoi(argv[13+i]);
    }
    
    NeuralNetwork NN(Layers,sizes);
    
    double cost_cv=0;    
    bool save_weights = atoi(argv[14+Layers]);
    bool load_weights = atoi(argv[15+Layers]);
    std::string weights_label = argv[20+Layers];
    
    if(load_weights)
    {
        //std::cout<<"loading weights and biases from: "<<weights_label<<std::endl;
        NN.loadWeights(weights_label);
    }
    
    
    bool train = atoi(argv[13+Layers]);
    if(train)
        NN.train(trainSize, eta, lambda, epochs, train_label_output, batchSize, 
                monitor_cost, print_cost, verbose, 
                monitor_accuracy, print_accuracy);
   
    
    if(save_weights)
    {
        //std::cout<<"saving weights and biases at: "<<weights_label<<std::endl;
        NN.saveWeights(weights_label);
    }
    
    
    bool test_NN = atoi(argv[16+Layers]);
    if(test_NN){
        bool cross_validate = atoi(argv[17+Layers]);
        int testSize = atoi(argv[18+Layers]);

        std::string test_label_output = argv[19+Layers];
        std::ofstream outfile;
        outfile.open(test_label_output, std::ios_base::app);
        double accuracy;
        if(cross_validate){
            accuracy = NN.test( testSize, im_train_name, label_train_name,true,cost_cv,batchSize,false, trainSize);
            std::cout<<"accuracy: "<<accuracy<<" cost: "<<cost_cv<<std::endl;
            outfile<<accuracy<<"\t"<<cost_cv<<std::endl;
        }
        else{
            accuracy = NN.test( testSize, im_test_name, label_test_name,true,cost_cv,batchSize,false, 0);
            std::cout<<"accuracy: "<<accuracy<<" cost: "<<cost_cv<<std::endl;
            outfile<<accuracy<<"\t"<<cost_cv<<std::endl;
        }
    }
    return 0;
}

//  //  learningCurve(Layers, sizes, outName,  eta,  lambda,  batchSize,  epochs, sizeStep,  maxSize, out_cost);

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