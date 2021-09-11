import scratchdiff.Unit as sd
from scratchdiff.Unit import Unit

w1 = Unit(2)
w2 = Unit(3)
w3 = sd.add(sd.sin(w1), sd.mul(w2, w1)) 
w3.backward_pass()
print(w1.grad.data)
print(w2.grad.data)
