import numpy
class Net(object):
    def __init__(self):
        self.inputLayerSize = 784
        self.hiddenLayerSize = 32
        self.outputLayerSize = 10

        self.W1 = numpy.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def training(self, X, y, runs, learningRate, sampleSize):
        for i in xrange(runs):
            rIndex = numpy.random.randint(0,X.shape[0],sampleSize)
            sampleX = X[rIndex]
            sampley = y[rIndex]
            dW1,dW2 = self.backprop(sampleX,sampley)
            self.W1 -= learningRate * dW1
            self.W2 -= learningRate * dW2
            print self.costFunction(X,y)

    def forward(self,X):
        self.z2 = numpy.mat(X) * numpy.mat(self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = numpy.mat(self.a2) * numpy.mat(self.W2)
        a3 = self.sigmoid(self.z3)
        return a3

    def backprop(self,X,y):
        self.yhat = self.forward(X)
        self.delta = numpy.multiply(-(y-self.yhat), self.sigmoidPrime(self.z3))
        dW2 = numpy.mat(self.a2.T) * numpy.mat(self.delta)
        dW1 = numpy.dot(X.T, numpy.multiply(numpy.dot(self.delta,self.W2.T) , self.sigmoidPrime(self.z2)) )
        return dW1, dW2

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        #create an
        self.yHat = self.forward(X)
        J = 0.5*numpy.sum(numpy.square(y-self.yHat))
        return J

    def sigmoid(self,z):
        return 1/(1+numpy.exp(-z))

    def sigmoidPrime(self,z):
        return numpy.multiply(self.sigmoid(z),(1-self.sigmoid(z)))
