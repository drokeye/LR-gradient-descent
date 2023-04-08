class LinearRegression:
    def __init__(self, X, Y, *, learning_rate=0.01) -> None:
        self.X = X
        self.Y = Y
        self.slope = 0
        self.intercept = 0
        self.learning_rate = learning_rate

    def apply_gradient_descent(self):
        Y_pred = self.hypothesis(self.X)
        Y = self.Y
        m = len(Y)
        self.slope = self.slope - (self.learning_rate * (1/m*(np.sum(self.X*(Y_pred - Y)))))
        self.intercept = self.intercept - (self.learning_rate * (1/m*(np.sum(Y_pred - Y))))

    def hypothesis(self, X=[]):
        Y_pred = []
        if len(X)==0:
            X = self.X
        for x in X:
            x = float(x)
            h = self.slope*x + self.intercept
            Y_pred.append(h)
        return np.array(Y_pred)
    
    def get_cost(self, Y_pred):
        m = len(self.X)
        return (1/2*m)*((np.sum(Y_pred - self.Y))**2)

    def plot(self, Y_pred, *, fig_name):
        f = plt.figure(fig_name)
        plt.scatter(self.X, self.Y, color='r')
        plt.plot(self.X, Y_pred, color='b')
        f.show()
