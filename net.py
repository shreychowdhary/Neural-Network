import numpy

class Net(object):
    def __init__(self,input,hidden,out):
        self.inputLayerSize = input
        self.hiddenLayerSize = hidden
        self.outputLayerSize = out

        self.W1 = numpy.random.randn(self.hiddenLayerSize,self.inputLayerSize+1)
        self.W2 = numpy.random.randn(self.outputLayerSize,self.hiddenLayerSize+1)

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


    def eval(self,X,y):
        res = numpy.argmax(self.forward(X),axis = 1)
        res = numpy.mat(res).T
        print res, numpy.sum(res == y)
        print numpy.sum(res == y)/float(y.shape[0])

    def forward(self,X):
        Xn = numpy.zeros((X.shape[0],X.shape[1]+1))
        Xn[:,1:] = X
        Xn[:,0] = 1
        self.a1 = Xn
        self.z2 = numpy.dot(self.a1,self.W1.T)
        self.a2 = self.sigmoid(self.z2)
        a2n = numpy.zeros((self.a2.shape[0],self.a2.shape[1]+1))
        a2n[:,1:] = self.a2
        a2n[:,0] = 1
        self.a2 = a2n
        self.z3 = numpy.dot(self.a2,self.W2.T)
        a3 = self.sigmoid(self.z3)
        return a3

    def backprop(self,X,y):
        self.yhat = self.forward(X)
        self.delta3 = -(y-self.yhat)
        dW2 = numpy.dot(self.delta3.T,self.a2)/X.shape[0]
        self.delta2 = numpy.multiply(numpy.dot(self.delta3,self.W2[:,1:]) , self.sigmoidPrime(self.z2))
        dW1 = numpy.dot(self.delta2.T,self.a1)/X.shape[0]
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

    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * (self.inputLayerSize + 1)
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + (self.hiddenLayerSize + 1)*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
