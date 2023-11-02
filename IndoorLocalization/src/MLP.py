import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from Network.MultiLayerPerceptron import MLP


class MLPNetwork():
    """Class for training MLP network
    """

    def __init__(self, trainDataSet, testDataSet) -> None:
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu"
                       )

        self.batchSize = 64
        self.trainLoader = DataLoader(
            trainDataSet, batch_size=self.batchSize, shuffle=True)
        self.testLoader = DataLoader(
            testDataSet, batch_size=self.batchSize, shuffle=True
        )
        self.learningRate = 0.1
        self.model = MLP().to(self.device)
        self.lossFunction = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.learningRate)

        self.bestModel = None
        self.bestAccracy = 0.0

    def train(self, dataloader, model, lossFunction, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = model(X)
            loss = lossFunction(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, model, lossFunction) -> float:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += lossFunction(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct

    def epochsLoop(self, epochs=50):
        for t in range(epochs):
            print(f"Epoch {t+1} / {epochs}\n-------------------------------")
            self.train(dataloader=self.trainLoader,
                       model=self.model,
                       lossFunction=self.lossFunction,
                       optimizer=self.optimizer)
            accuracy = self.test(dataloader=self.testLoader,
                                 model=self.model,
                                 lossFunction=self.lossFunction)
            if accuracy > self.bestAccracy:
                self.bestAccracy = accuracy
                self.bestModel = self.model
        torch.save(self.bestModel.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")
        print("Done!")
