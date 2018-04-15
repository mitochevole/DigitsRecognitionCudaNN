# DigitsRecognitionCudaNN

A simple C++/CUDA project for me to understand neural networks.
It is inspired by the book on [neural networks and deep learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen and on the practical excercises of weeks three and four of the [Coursera Machine Learning course](https://www.coursera.org/learn/machine-learning) by Andrew Ng.


I have never received any proper programming training so the code might lack best practices. Apologies for that. Hopefully, it is simple and clear enough to understand for those willing to.

### Why ###

My main goal was to work on my own on stochastic gradient descent training of neural network, understanding feedforward and backpropagation algorithms and the role of hyperparameters. Therefore, the implementation is naïve and probably not the most efficient. I am not using any linear algebra or NN cuda libraries. 

The program reads training and test data from the [MNIST handwritten digits database](http://yann.lecun.com/exdb/mnist/) and reads them in using an adapted version of this [c++ code for reading MNIST data-set](https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set).

### How ###

The core of the code is the `NeuralNetwork` class, which is built on top of the `d_matrix` class. Most of the computation takes place on GPU, exploiting parallelization of matrix operations, the auxiliary `matrix`class is used as a CPU buffer to display, save and load data from and to the GPU.

The `NeuralNetwork` class is initialised by specifying the number of layers and the number of nodes for each layer.
This will generate the weights and biases matrices. The class is equipped with the random initialisation function and feedforward, backpropagation and gradient descent algorithms that do not depend on the specific problem. This means that in the future I could work on a more general version of the code that can behave as a more generic classifier. However, for the time being, loading of the data-set and testing only works with MNIST 28x28 pixels images.






