from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import f1_score


class OneVsOneSVM:
    def __init__(self, xTrain, yTrain, xTest, yTest) -> None:
        self.module = SVC(
            kernel='linear', decision_function_shape='ovo', verbose=True)
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest

    def train(self):
        self.module.fit(self.xTrain, self.yTrain)

    def test(self):
        yPred = self.module.predict(self.xTest)
        accuracy = accuracy_score(self.yTest, yPred)
        f1Score = f1_score(self.yTest, yPred, average="macro")
        confusionMatrix = confusion_matrix(self.yTest, yPred)
        print(f"Accrucy: {accuracy:.8f}\n")
        print(f"F1 Score: {f1Score:.8f}\n")
        print(f"Confusion Matrix: \n{confusionMatrix}\n")
