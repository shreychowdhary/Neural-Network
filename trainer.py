from scipy import optimize
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='TNC', \
                                 args=(X, y), options={'maxiter': 10000, 'disp' : True}, callback=self.callbackF,)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
