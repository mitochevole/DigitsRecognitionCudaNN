/* 
 * File:   utilities.h
 * Author: michele
 *
 * Created on 09 April 2018, 14:33
 */

#ifndef UTILITIES_H
#define	UTILITIES_H

#include "NeuralNetwork.h"


/************TESTING NN***************/
//prints learning curve for a neural network with a given topology,
//hyper-parameters can be changed





void learningCurve(int Layers, int * sizes, std::string outName, 
        double eta, double lambda, int batchSize, int epochs, 
        int sizeStep, int maxSize, bool out_cost){
    
    std::ofstream output;
    output.open(outName.c_str());  
    for(int size = sizeStep; size < maxSize; size+=sizeStep){
        const int Layers = 3;
        int sizes[Layers] = {im_size,30,10};
        NeuralNetwork NN(Layers,sizes);
        double cost_train = 0, cost_cv = 0;
        NN.train(size, eta, lambda, epochs, batchSize,false, false, false, false);
        std::cout<<"training complete"<<std::endl;
        double testOnTrain = NN.test( size, im_train_name, label_train_name,true,cost_train ,batchSize,false, 0);
        std::cout<<"test on train set complete"<<std::endl;
        double CV = NN.test( size, im_train_name, label_train_name,true,cost_cv,batchSize,false, maxSize);
        std::cout<<"test on CV set complete"<<std::endl;
        output<<size<<"\t"<<testOnTrain<<"\t"<<CV<<"\t"<<cost_train<<"\t"<<cost_cv<<std::endl;   
    }    
    output.close();
}


//void learning_over_epochs(std::string outName, 
//        double eta, double lambda, int batchSize, int size, int maxEpochs, int stepEpochs){
//    
//    std::ofstream output;
//    output.open(outName.c_str());  
//    for(int ep = stepEpochs; ep < maxEpochs; ep+=stepEpochs){
//        const int Layers = 3;
//        int sizes[Layers] = {im_size,30,10};
//        NeuralNetwork NN(Layers,sizes);
//        double cost_train =0, cost_cv = 0;
//        NN.train(size, eta, lambda, ep, batchSize,false, false);
//        std::cout<<"training complete"<<std::endl;
//        double testOnTrain = NN.test( size, im_train_name, label_train_name,true,cost_train ,1,false, 0);
//        std::cout<<"test on train set complete"<<std::endl;
//        double CV = NN.test( size, im_train_name, label_train_name,true,cost_cv,1,false, size);
//        std::cout<<"test on CV set complete"<<std::endl;
//        output<<ep<<"\t"<<testOnTrain<<"\t"<<CV<<"\t"<<cost_train<<"\t"<<cost_cv<<std::endl;   
//    }    
//    output.close();
//}

#endif	/* UTILITIES_H */

