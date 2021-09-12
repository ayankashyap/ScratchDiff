import scratchdiff.Unit as sd
from scratchdiff.Unit import Unit

# sin(x) + x*y
w1 = Unit(2)
w2 = Unit(3)
w3 = sd.add(sd.sin(w1), sd.mul(w2, w1)) 
w3.backward_pass()
print(w1)
print(w2)

# sin(x)/cos(x) = tan(x)

x = Unit(30)
x1 = sd.sin(x)
x2 = sd.cos(x)
x3 = sd.div(x1,x2)
x3.backward_pass()
print(x)
