import numpy
class Net(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        self.W1 = numpy.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self,X):
        self.z2 = numpy.mat(self.W1) * numpy.mat(X)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = numpy.mat(self.W2) * numpy.mat(a2)
        self.a3 = self.sigmoid(self.z3)
        return a3

    def sigmoid(self,z):
        return 1/(1+numpy.exp(-z))
