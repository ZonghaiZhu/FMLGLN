# FMLGLN: Fast Multi-layer Graph Linear Network

## Overview

This work proposes a Fast Multi-layer Graph Linear Network (FMLGLN) with a straightforward structure and a low amount of hyperparameters to deal with large-scale graph data. In the implementation, FMLGLN raises the normalized adjacency matrix to the power of different values and multiplies these matrices with original features to get embedding features of nodes in pre-processing. Finally, FMLGLN concatenates these embedding features and uses a linear neural network to conduct training. The matrix multiplication is linear, and the network structure consists of linear layers, bringing a fast training speed. Experimental results on large-scale graph-structured data demonstrate the effectiveness of the proposed FMLGLN.

##

The documents include FMLGLN_papers100M_trn, FMLGLN_papers100M_trn_all_adj, FMLGLN_papers100M_val, FMLGLN_papers100M_val_all_adj, FMLGLN_Trn, and FMLGLN_val. 'trn' related documents correspond to the training process and 'val' related document corresponds to the validation process providing the best combination of hyper-parameters. For 'papers100M' related documents, 'all_adj' means use that whole adjacency matrix to obtain the embedding features as SIGN does, while the rest are trained under the proposed inductive way in this work.


## Results

We have provided all classification results in the uploaded codes. Please check the csv files.

All results can be reproduced by running main.py. In main.py, you can select the dataset and adjust the parameters. The main.py will use model.py, which contains the fit function and predict function. We have used the grid search to conduct the experiment and record the results on validation and test data. The parameters combination that achieves the highest results on the validation data is selected to predict the test data. 


## Dependencies

* python >= 3.7.6
* pytorch >= 1.6.0
* torch_geometric >= 1.6.1
* numpy >= 1.19.2
* scikit-learn >= 0.23.1
* scipy >= 1.41
* torch_sparse >= 0.6.7
* ogb >= 1.2.3

## Run Program

* main.py calls the main function to run the program
* model.py represents the model and contains training and testing
* network.py is the structure of the used neural network
* utils.py provides some tools to assist program execution 
* dataset.py provides the code about how the get the dataloader of training, validation and test nodes.

Run main.py directly to get the results on validation and test data. You can change the parameters and choose different data set in argparse.ArgumentParser() to see different results. 

## Datasets

All datasets used in this work can be downloaded from "https://github.com/GraphSAINT/GraphSAINT" provided in the paper "GraphSAINT: Graph Sampling Based Inductive Learning Method". The ogbn-papers100M can be downloaded form ogb datasets.

## Customization

### How to Prepare Your Own Dataset?

You can prepare the graph data with adjacency matrix A, feature matrix X, and corresponding label Y. Our FMLGLN mainly conduct linear operations on the adjacency matrix A. Change the corresponding position of A, X , and Y in the code.



