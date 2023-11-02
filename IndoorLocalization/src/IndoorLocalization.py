import torch
from torch.utils.data import TensorDataset
from DataLoader.DataSplit import getTrainTestSplit
from SVM import OneVsOneSVM
from MLP import MLPNetwork

[xTrain, xTest, yTrain, yTest] = getTrainTestSplit(0.33)

svm = OneVsOneSVM(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)
svm.train()
svm.test()

xTrain = torch.LongTensor(xTrain)
yTrain = torch.LongTensor(yTrain).add(-1)
xTest = torch.LongTensor(xTest)
yTest = torch.LongTensor(yTest).add(-1)
trainDataset = TensorDataset(xTrain, yTrain)
testDataset = TensorDataset(xTest, yTest)
mlp = MLPNetwork(trainDataset, testDataset)
mlp.epochsLoop(100)
