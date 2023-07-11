import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
class NeuralNetwork:

    # m: number train
    # n0: features 
    # n1: number of neurons of layer _1
    # n2: number of neurons of layer _2

    def __init__(self, nInput, nOutput, hiddenNxM, rand_seed = 0) -> None :

        # pseudo-random (fixed the same random generation)
        np.random.seed(rand_seed)

        self.params = {}
        size_layers = [nInput] + hiddenNxM  + [nOutput]

        # Initialization on Layer k
        for k in range(1, len(size_layers)):
            self.params['W_' + str(k)] = np.random.randn(size_layers[k], size_layers[k - 1])
            self.params['B_' + str(k)] = np.random.randn(size_layers[k], 1)

        print("\n\t Number of Neurons for each layer \n\t", size_layers, "\n")

    def getParams(self):
        return self.params
    
    def log_loss(self, A, Y):
        epsilon = 1e-15
        LL = -1 / len(Y) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        return LL

    def forward_prop(self, X):

        self.A = {'A_0': X}
        nLayers = len(self.params) // 2 # total number of layers

        for k in range(1, nLayers + 1):
            # Z[k] = <W[k] | A[k-1]> + B[k]
            Z = np.dot(self.params['W_' + str(k)], self.A['A_' + str(k - 1)]) +  self.params['B_' + str(k)]
            Z = np.clip(Z, -700, 700)
            # A[k] = f(Z[k])
            self.A['A_' + str(k)] = 1 / (1 + np.exp(-Z))

        return self.A
    
    def back_prop(self, Y):
        
        m = Y.shape[1]
        nLayers = len(self.params) // 2 # total number of layers

        # dgradZ[k] = Y[k] - A[k]  k: last layer
        dgradZ_k = Y - self.A['A_' + str(nLayers)] 
        self.grad = {}

        for k in reversed(range(1, nLayers + 1)):
            self.grad['gradW_' + str(k)] = -1/m * np.dot(dgradZ_k, self.A['A_' + str(k - 1)].T)
            self.grad['gradB_' + str(k)] = -1/m * np.sum(dgradZ_k, axis=1, keepdims=True)
           
            # update
            if k > 1:

            # dgradZ[k-1] = <W[k] | dgradZ[k]> * A[k-1] * (1 -A[k-1])
                dgradZ_k = np.dot(self.params['W_' + str(k)].T, dgradZ_k) * self.A['A_' + str(k)] * (1 - self.A['A_' + str(k - 1)])

        return self.grad
    
    def update(self, learning_rate):
        
        nLayers = len(self.params) // 2 # total number of layers
        
        for k in range(1, nLayers + 1):
            self.params['W_' + str(k)] = self.params['W_' + str(k)] - learning_rate * self.grad['gradW_' + str(k)]
            self.params['B_' + str(k)] = self.params['B_' + str(k)] - learning_rate * self.grad['gradB_' + str(k)]

        return self.params
    
    def predict(self, X):
        A = self.forward_prop(X)
        last_layer = len(self.params) // 2
        return A['A_' + str(last_layer)] >= 0.5
    
    def calibrate(self, X_train, Y_train, X_test, Y_test, learning_rate=0.01, iteration = 1000, n_sync=10):

        train_Loss = []
        train_Acc = []
        test_Loss = []
        test_Acc = []

        for i in tqdm(range(iteration)):

            A_train = self.forward_prop(X_train)     
            self.back_prop(Y_train)
            self.update(learning_rate)

            if i % n_sync == 0 :

                last_layer = len(self.params) // 2

                # train
                train_Loss.append(self.log_loss(A_train['A_' + str(last_layer)], Y_train))
                Y_pred = self.predict(X_train)
                train_Acc.append(accuracy_score(Y_train.flatten(), Y_pred.flatten()))

                # test
                A_test = self.forward_prop(X_test)
                test_Loss.append(self.log_loss(A_test['A_' + str(last_layer)], Y_test))
                Y_pred = self.predict(X_test)
                test_Acc.append(accuracy_score(Y_test.flatten(), Y_pred.flatten()))

        return {
            'train_Loss': train_Loss,
            'train_Acc': train_Acc,
            'test_Loss': test_Loss,
            'test_Acc': test_Acc
        }
    

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

