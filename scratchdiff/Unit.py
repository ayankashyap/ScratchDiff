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

    def __repr__(self) -> str:
        return f"<Unit {self.data!r} with grad {(self.grad.data if self.grad else None)!r} and grad_fn=\"{(self.derivative.__name__ if self.derivative.__name__ != '<lambda>'  else None)}\">"

    def zero_grad(self):
        self.grad = Unit(
            np.zeros_like(self.data, dtype=np.float64), requires_grad=False
        )

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
        if self.shape != ():
            raise RuntimeError(
                "cannot run backward pass on non-scalar outputs. Try and use a reducing function like sum()"
            )
        computation_order = reversed(self._deepwalk())
        self.grad = Unit(1, requires_grad=False)
        for n in computation_order:
            n.derivative()


def sin(t):
    t = t if isinstance(t, Unit) else Unit(t)
    out = Unit(np.sin(t.data), children=[t])

    def _sin_Backward():
        if t.requires_grad:
            t.grad.data += np.cos(t.data) * out.grad.data

    out.derivative = _sin_Backward
    return out


def cos(t):
    t = t if isinstance(t, Unit) else Unit(t)
    out = Unit(np.cos(t.data), children=[t])

    def _cos_Backward():
        if t.requires_grad:
            t.grad.data += -np.sin(t.data) * out.grad.data

    out.derivative = _cos_Backward
    return out


def power(t, n):
    t = t if isinstance(t, Unit) else Unit(t)
    pow_exp = "*".join(["t.data"] * n)
    out = Unit(eval(pow_exp), children=[t])

    def _power_Backward():
        if t.requires_grad:
            t.grad.data += n * (t.data ** (n - 1)) * out.grad.data

    out.derivative = _power_Backward
    return out


def add(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    out = Unit(np.add(t1.data, t2.data), children=[t1, t2])

    def _add_Backward():
        if t1.requires_grad:
            t1.grad.data += out.grad.data
        if t2.requires_grad:
            t2.grad.data += out.grad.data

    out.derivative = _add_Backward
    return out


def mul(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    out = Unit(np.multiply(t1.data, t2.data), children=[t1, t2])

    def _mul_Backward():
        if t1.requires_grad:
            t1.grad.data += t2.data * out.grad.data
        if t2.requires_grad:
            t2.grad.data += t1.data * out.grad.data

    out.derivative = _mul_Backward
    return out


def div(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    out = Unit(np.divide(t1.data, t2.data), children=[t1, t2])

    def _div_Backward():
        if t1.requires_grad:
            t1.grad.data += (t2.data ** -1) * out.grad.data
        if t2.requires_grad:
            t2.grad.data += -(t1.data * t2.data ** -2) * out.grad.data

    out.derivative = _div_Backward
    return out


def matmul(t1, t2):
    t1 = t1 if isinstance(t1, Unit) else Unit(t1)
    t2 = t2 if isinstance(t2, Unit) else Unit(t2)
    if t1.shape[-1] != t2.shape[0]:
        raise ValueError(f"Shapes dont align: {t1.shape[-1]}!={t2.shape[0]}")

    out = Unit(t1.data @ t2.data, children=[t1, t2])

    def _matmul_Backward():
        # grad.data is always in the form of a vector/matrix of ones in the shape of AB,
        #  since we always have to do a reduction of the output tensor
        if t1.requires_grad:
            # matrix mul: C = AB , grad(C)w.r.t A = B^Transpose
            if len(out.grad.shape)>1:
                t1.grad.data += np.swapaxes(out.grad.data, -2, -1) @ t2.data.T
        if t2.requires_grad:
            # matrix mul: C = AB , grad(C)w.r.t B = A
            print(t1.shape)
            print(out.grad.shape)
            t2.grad.data += t1.data @ out.grad.data

    out.derivative = _matmul_Backward
    return out


def sum(t):
    t = t if isinstance(t, Unit) else Unit(t)
    out = Unit(np.sum(t.data), children=[t])

    def _sum_Backward():
        if t.requires_grad:
            print(out.grad.data)
            t.grad.data += np.ones_like(t.data) * out.grad.data

    out.derivative = _sum_Backward
    return out


