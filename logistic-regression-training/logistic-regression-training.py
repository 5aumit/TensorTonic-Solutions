import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    
def _bceloss(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Lengths not matching"
    n = len(y_true)
    return y_true*(np.log(y_pred)) + (1-y_true)*np.log((1-y_pred))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)

    #Initialize weights & biases
    w = np.zeros((X.shape[1]))
    b = np.array([0])

    for iter in range(steps):
        print(w,b)
        #Forward pass
        z = X@w + b
        #Activation
        p = np.squeeze(_sigmoid(z))

        #Loss
        # loss = _bceloss(y,p)

        #Gradients
        dw = (1/n)*(X.T@(p-y))
        db = (1/n)*np.sum(p-y)

        #Update weights & biases
        w = w-lr*dw
        b = b-lr*db

    return w,b