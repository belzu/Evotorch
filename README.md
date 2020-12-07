# Evotorch
Evotorch is a neuro-evolution library written in Python that makes use of the Pytorch library formalism.
It allows the evolution of multilayer and convolutional networks.
This project was conceived as my end-of-degree project at UPV/EHU university.

Evotorch uses genetic algorithms (a paradigm of evolutionary algorithms) in order to obtain a neural network that contains the best combination of values
for a user-elected set of hiperparameters so that a given problem (in the form of a CSV file) can be solved.

## Modules

### Network

This module is responsible for implementing a neural network using Pytorch classes, functions and data. It allows the user to create a neural network from the data
input, as well as to save or load one. Evotorch allows you to create two different neural networks: **MLP** (MultiLayer Perceptron) and **CNN** (Convolutional Neural Network).

## GenAlgs

This module is in charge of implementing a genetic algorithm using DEAP classes, functions and data. Allows the user to create a genetic algorithm.
Evotorch allows you to create two different types of genetic algorithms: GenAlg_MLP and GenAlg_CNN.
The user is allowed to decide which set of hyperparameters can evolve. For MLP networks are: the number of neurons in each hidden layer, the learning rate, the
number of epochs, patience, number of hidden layers, activation functions and the dropouts. For CNN networks, in addition to the same as MLP networks, they are: number of convolutional sequences, the number of kernels in each convolutional layer, the size of the kernels in each convolution and pooling layer and the size of the stride in each convolution layer.

A convolutional sequence is, as the name suggests, a sequence of two layers: convolutional layers + MaxPooling layers.

## Executing Evotorch

Executing Evotorch is very simple, but there are two things that the user must take into account: with Evotorch you can either evolve neural network or execute them. These concepts aren't excluying, on the contrary, they are complementary!
* In the **exec_MLP.py** and **exec_CNN.py** files examples on how to execute the neural networks are found. These files were created with the purpose of testing the execution of the neural networks.
* In the **evolve_MLP.py** and **evolve_CNN.py** files examples on how to obtain optimized neural networks are found. These files were created with the purpose of testing the genetic algorithms.

**By default Evotorch DOES NOT return a Neural Network, it returns THE OPTIMAL COMBINATION OF VALUES FOR THE HIPERPARAMETERS THAT WERE ELECTED BY THE USER; in other words, it returns the combination of hiperparameter values that results in the optimal learning of a neural network for a given problem.**

This combination of hiperparameters is represented as a text file containing the following format:
* For MLP networks:
  * list of integers
  * integer
  * integer
  * list of strings or string
  * list of floats
  * integer
  * integer
  * float
  * string
  * string
  * integer
  * integer
  * string

* For CNN networks:
  * list of integers
  * integer
  * integer
  * list of strings or string
  * list of floats
  * integer
  * integer
  * float
  * string
  * string
  * integer
  * integer
  * list of integers
  * list of list of two integers
  * list of integers
  
### Steps for properly execute Evotorch

1. Define the hiperparameters' values as well as which ones you want to evolve along with the evolution's parametric criteria.
2. Create an instance of a genetic algorithm by calling GenAlg_MLP(GenAlg_MLP_Descriptor(< arguments >)) or GenAlg_CNN(GenAlg_CNN_Descriptor(< arguments >)).
3. Execute *simple_genetic_algorithm function*. A file named NN_< datetime >.txt will appear in the directory you choose (./ by default) containing the optimal values for the hiperparameters.
4. Create an instance of a neural network with MLP_Network(MLP_Descriptor()) or CNN_Network(CNN_Descriptor()).
5. Pass the hiperparameters you obtained on the step **3** to the neural network you created on the step 4 with *load_NN_info(< text file containing the hiperparameters >)* function.
6. Execute the *training_MLP(< training set >)* and *testing_MLP(< test set >)* functions. You will obtain the accuracy or MSE loss (depending on the kind of problem we are facing: classification (MLP networks with mode = "clssf" and CNN) or regression (MLP networks with mode = "rgrss")). You can save the neural network with *save_NN( < file name >)* function and load it with load_NN(*< file name >)*.
