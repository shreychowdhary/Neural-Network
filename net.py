import numpy
class Net(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        self.W1 = numpy.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self,X):
        self.z2 = numpy.mat(X) * numpy.mat(self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = numpy.mat(self.a2) * numpy.mat(self.W2)
        a3 = self.sigmoid(self.z3)
        return a3



    def backprop(self,X,y):
        self.yhat = self.forward(X)
        delta = numpy.multiply(-(y-self.yhat), self.sigmoidPrime(self.z3))
        dW2 = numpy.mat(self.a2.T) * numpy.mat(delta)
        dW1 = numpy.mat(X.T) * ((numpy.mat(delta) * numpy.mat(self.W2.T)) * self.sigmoidPrime(self.z2)) 
        return dW2, dW1

    def sigmoid(self,z):
        return 1/(1+numpy.exp(-z))

    def sigmoidPrime(self,z):
        return numpy.multiply(self.sigmoid(z),(1-self.sigmoid(z)))
