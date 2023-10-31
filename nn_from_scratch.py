import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1330)

def one_hot(y):
    one_hot_y = np.zeros(10)
    one_hot_y[y] = 1
    return one_hot_y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.exp(z).sum()

def CEloss(pred, y):
    return -np.dot(y, np.log(pred))

class feedforward():
    def __init__(self, d, d1, k, lr=0.0003, num_iters=1000):
        self.d = d
        self.d1 = d1
        self.k = k
        self.lr = lr
        self.num_iters = num_iters
        self.train_error = []
        self.test_error = []

    def init_params_to_zero(self):
        self.w1 = np.zeros((self.d1, self.d))
        self.w2 = np.zeros((self.k, self.d1))

    def init_params_random_normal(self):
        self.w1 = np.random.normal(scale=1., size=(self.d1, self.d))
        self.w2 = np.random.normal(scale=1., size=(self.k, self.d1))

    def init_params_b(self):
        self.w1 = np.random.uniform(low=-1., high=1., size=(self.d1, self.d))
        self.w2 = np.random.uniform(low=-1., high=1., size=(self.k, self.d1))

    def forward(self, x):
        self.h1 = np.dot(self.w1, x)
        self.a1 = sigmoid(self.h1)
        self.h2 = np.dot(self.w2, self.a1)
        self.y_pred = softmax(self.h2)
        return self.y_pred

    def backward(self, x, y):
        self.h2_grads = self.y_pred - y
        self.w2_grads = np.outer(self.h2_grads, self.a1.T)
        self.h1_grads = np.matmul(self.w2.T, self.h2_grads) * self.a1 * (1-self.a1)
        self.w1_grads = np.outer(self.h1_grads, x.T)

    def update_network(self):
        self.w1 -= self.lr * self.w1_grads
        self.w2 -= self.lr * self.w2_grads

    def eval(self, X, Y):
        errors = 0
        for x, y in zip(X, Y):
            errors += int(np.argmax(self.forward(x))!=y)
        test_error = errors / len(Y)
        self.test_error.append(test_error)
        print("Test Error:", test_error)

    def train(self, X, Y, X_eval, Y_eval):
        self.init_params_b()
        for iter in range(self.num_iters):
            print('running iteration:', iter)
            train_loss = 0
            for x, y in zip(X, Y):    
                y = one_hot(y)
                pred = self.forward(x)
                train_loss += CEloss(pred, y)
                self.backward(x, y)
                self.update_network()
            
            print('training loss:', train_loss)
            self.train_error.append(train_loss)
            if iter % 1 == 0:
                self.eval(X_eval, Y_eval)
        
        plt.plot([iter for iter in range(len(self.test_error))], self.test_error)
        plt.xlabel('iters')
        plt.ylabel('test error')
        plt.savefig('./b_test_error.pdf')
        plt.clf()

        plt.plot([iter for iter in range(len(self.train_error))], self.train_error)
        plt.xlabel('iters')
        plt.ylabel('train error')
        plt.savefig('./b_train_error.pdf')
        plt.clf()
        




if __name__ == '__main__':
    net = feedforward(784, 200, 10, num_iters=20)

    dataset = np.array(pd.read_csv('./mnist/train.csv'))
    X, Y = dataset[:, 1:], dataset[:, 0]
    X_eval, Y_eval = X[:1000], Y[:1000]
    X_train, Y_train = X[1000:], Y[1000:]

    net.train(X_train, Y_train, X_eval, Y_eval)

