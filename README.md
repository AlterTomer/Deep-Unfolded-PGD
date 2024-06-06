# Deep-Unfolded-PGD
Python repository for the paper "Rapid Optimization of Superposition Codes for
Multi-Hop NOMA MANETs via Deep Unfolding"
---
## Table of Contents
 - [Introduction](#introduction)
 - [PGDNet (python codes)](#PGDNet (python codes))
 - [GNN (python codes)](#GNN (python codes))
---
## Introduction
This repository implements the proposed method of PGDNet for NOMA MANET, a deep-unfolded deep learning power allocation optimization algorithm.
We also implement a graph neural network (GNN) algorithm which serves as a benchmark for the PGDNet.


T. Alter and N. Shlezinger, "Rapid Optimization of Superposition Codes for
Multi-Hop NOMA MANETs via Deep Unfolding".

---
## PGDNet
Includes the network module, and all needed functions for net analysis, channel generation, and plotting functions.
To run an analysis run the GeneralNetAnalysis.py code after adjusting the parameters to match the MANET's topology.
Change the dir_path variable to the path you would like to save the model into.

---
## GNN
Includes the network module, and all needed functions for net analysis, channel generation, and plotting functions.
To run an analysis run the GNN_test.py code after adjusting the parameters to match the MANET's topology.
Change the dir_path variable to the path you would like to save the model into.
