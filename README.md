# SCRATCH-DIFF : implementation of autodifferentiation in python

This repo does not contain any clever or fancy implementation of autodifferentiation. Its heavily inspired by older blogs and repos. The code is heavily simplified and not at all optimized XD. The primary goal is to learn and understand how it really works. Currently, It only has one tensor class called "Unit" with a functioning autograd engine that supports a few functions.
The goal of this project was to learn to implement autodifferentiation and nothing more. 

## Some Examples
```
from scratchdiff.Unit import *
# sin(x) + x*y
w1 = Unit(2)
w2 = Unit(3)
w3 = add(sin(w1), mul(w2, w1)) 
w3.backward_pass()
print(w1.grad.data)
print(w2.grad.data)

```

```
# z = w.x + b

w = Unit([2, 2, 2])
x = Unit([1, 1, 1])
b =Unit([1,1,1])
z = sum(add(dot(w, x), b))
z.backward_pass()
print(w)
print(x)
print(b)

```
### Basic MSE Loss and gradient descent for regression
```
from sklearn.datasets import make_regression
X,Y= make_regression(n_samples=200, n_features=1, noise=50,random_state=3)
def MSE(y,yhat):
    N = 1/y.data.shape[0]
    a = add(yhat , -y.data)
    s = sum(power(a, 2))
    loss =  mul(s, Unit(N, requires_grad=False))
    return loss

X = Unit(X, requires_grad=False)
Y = Unit(Y.reshape(Y.shape[0], 1), requires_grad=False)

b = Unit(np.zeros((X.shape[0],1)))
W = Unit(np.random.normal(size=(X.shape[1], 1)))

for i in range(10000):
    W.zero_grad()
    b.zero_grad()
    yhat = add(matmul(X,W), b)
    loss = MSE(Y, yhat)
    print(loss)
    loss.backward_pass()
    W.data -= 0.001*W.grad.data
    b.data -= 0.001*b.grad.data

y_true = [i[0] for i in Y.data]
y_pred = [i[0] for i in yhat.data]
```

### TODOS in the future if I ever come back to this

- [ ] Write proper backwards pass functions for add, the current isn't robust to matrix shape

- [ ] classes for linear, conv, optimizers, etc

- [ ] complete mnist
