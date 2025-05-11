import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class lstm_cell:
    def __init__(self, i_s, h_s):
        self.i_s = i_s
        self.h_s = h_s
        

        # forget gate
        self.Wf = np.random.randn(h_s, h_s + i_s)
        self.bf = np.random.rand(h_s)

        # input gate
        self.Wi = np.random.randn(h_s, h_s + i_s)
        self.bi = np.random.rand(h_s)

        # output gate
        self.Wo = np.random.randn(h_s, h_s + i_s) 
        self.bo = np.random.rand(h_s)

        # cell gate
        self.Wc = np.random.randn(h_s, h_s + i_s)
        self.bc = np.random.rand(h_s)


    def forward(self, h_prev, c_prev, x):
        z = np.concatenate((h_prev, x))
        f = sigmoid(np.dot(self.Wf, z) + self.bf)
        ct = np.tanh(np.dot(self.Wc, z) + self.bc)
        ot = sigmoid(np.dot(self.Wo, z) + self.bo)
        c = f * c_prev + (1 - f) * ct
        h = ot * np.tanh(c)
        return c, h

        
op = lstm_cell(10, 20).forward(np.random.randn(20), np.random.randn(20), np.random.randn(10))
print(op[0].shape)