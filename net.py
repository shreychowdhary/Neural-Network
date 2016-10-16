import numpy
class Net(object):
    def __init__(self,input,hidden,out):
        self.inputLayerSize = input
        self.hiddenLayerSize = hidden
        self.outputLayerSize = out

        self.W1 = numpy.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        self.B1 = numpy.zeros((1,hidden))
        self.B2 = numpy.zeros((1,out))

    def train(self, X, y, runs, learningRate, sampleSize):
        print self.costFunction(X,y)
        for i in xrange(runs):
            rIndex = numpy.random.randint(0,X.shape[0],sampleSize)
            sampleX = X[rIndex]
            sampley = y[rIndex]
            dW1,dW2,dB1,dB2 = self.backprop(sampleX,sampley)
            self.W1 -= learningRate * dW1
            self.W2 -= learningRate * dW2
            self.B1 -= learningRate * dB1
            self.B2 -= learningRate * dB2
        print self.costFunction(X,y)

    def evaluate(self,X,y):
        res = numpy.argmax(self.forward(X),axis = 1)
        print (float(numpy.sum(res == y))/y.shape[0])

    def forward(self,X):
        self.z2 = numpy.mat(X) * numpy.mat(self.W1) + self.B1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = numpy.mat(self.a2) * numpy.mat(self.W2) + self.B2
        a3 = self.sigmoid(self.z3)
        return a3

    def backprop(self,X,y):
        self.yhat = self.forward(X)
        self.delta3 = numpy.multiply(-(y-self.yhat), self.sigmoidPrime(self.z3))
        dW2 = numpy.dot(self.a2.T,self.delta3)
        dB2 = numpy.sum(self.delta3, axis = 0, keepdims = true)
        self.delta2 = numpy.multiply(numpy.dot(self.delta3,self.W2.T) , self.sigmoidPrime(self.z2))
        dW1 = numpy.dot(X.T,self.delta2)
        dB1 = numpy.sum(self.delta2,axis = 0)
        return dW1, dW2, dB1, dB2

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
