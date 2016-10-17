import numpy
class Net(object):
    def __init__(self,input,hidden,out):
        self.inputLayerSize = input
        self.hiddenLayerSize = hidden
        self.outputLayerSize = out

        self.W1 = numpy.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = numpy.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def train(self, X, y, runs, learningRate, sampleSize):

        for i in xrange(runs):
            indexes = numpy.arange(X.shape[0])
            numpy.random.shuffle(indexes)
            X = X[indexes]
            y = y[indexes]
            for j in xrange(0,X.shape[0],sampleSize):
                dW1,dW2 = self.backprop(X[j:j+sampleSize],y[j:j+sampleSize])
                self.W1 -= learningRate * dW1
                self.W2 -= learningRate * dW2
            print self.costFunction(X,y)


    def evaluate(self,X,y):
        res = numpy.argmax(self.forward(X),axis = 1)
        print (float(numpy.sum(res == y))/y.shape[0])

    def forward(self,X):
        self.z2 = numpy.mat(X) * numpy.mat(self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = numpy.mat(self.a2) * numpy.mat(self.W2)
        a3 = self.tanh(self.z3)
        return a3

    def backprop(self,X,y):
        self.yhat = self.forward(X)
        self.delta3 = numpy.multiply(-(y-self.yhat), self.tanhPrime(self.z3))
        dW2 = numpy.dot(self.a2.T,self.delta3)
        self.delta2 = numpy.multiply(numpy.dot(self.delta3,self.W2.T) , self.tanhPrime(self.z2))
        dW1 = numpy.dot(X.T,self.delta2)
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

    def tanh(self,z):
        return numpy.tanh(z)

    def tanhPrime(self,z):
        return 1.0 - numpy.square(numpy.tanh(z))
