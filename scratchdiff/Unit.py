import numpy as np

class Unit:
    def __init__(self, data, requires_grad=True, children=[]):
        """Main computation class in the engine"""
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad, self.requires_grad = None, requires_grad
        self.children = children
        self.derivative = lambda: None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = Unit(np.zeros_like(self.data, dtype=np.float64), requires_grad=False)

    @property
    def shape(self):
        return self.data.shape

    def _deepwalk(self):
        def __deepwalk(node, visited, nodes):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    __deepwalk(child, visited, nodes)
                nodes.append(node)
            return nodes
        return __deepwalk(self, visited=set(), nodes=[])

    def backward_pass(self):
        computation_order = reversed(self._deepwalk())
        if self.shape == ():
            self.grad = Unit(1)
        for n in computation_order:
            n.derivative()

def sin(t):
    t = t if isinstance(t, Unit) else Unit(t)
    out = Unit(np.sin(t.data), children=[t])
    def _derivative(): 
        if t.requires_grad: t.grad.data += np.cos(t.data) * out.grad.data
    out.derivative = _derivative 
    return out

def cos(t):
    t = t if isinstance(t, Unit) else Unit(t)
    out = Unit(np.cos(t.data), children=[t])
    def _derivative(): 
        if t.requires_grad: t.grad.data += -np.sin(t.data) * out.grad.data
    out.derivative = _derivative 
    return out


def power(t, n):
    t = t if isinstance(t, Unit) else Unit(t)
    pow_exp = '*'.join(['t.data']*n)
    out = Unit(eval(pow_exp), children=[t])
    def _derivative(): 
        if t.requires_grad: t.grad.data += n * t.data * out.grad.data
    out.derivative = _derivative
    return out

def add(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    out = Unit(np.add(t1.data, t2.data), children=[t1, t2])
    def _derivative():
        if t1.requires_grad: t1.grad.data+=out.grad.data
        if t2.requires_grad: t2.grad.data+=out.grad.data
    out.derivative = _derivative
    return out 

def mul(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    out = Unit(np.multiply(t1.data, t2.data), children=[t1, t2])
    def _derivative():
        if t1.requires_grad: t1.grad.data+=t2.data*out.grad.data
        if t2.requires_grad: t2.grad.data+=t1.data*out.grad.data
    out.derivative = _derivative
    return out    


