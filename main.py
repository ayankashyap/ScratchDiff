from scratchdiff.Unit import *

# sin(x) + x*y
w1 = Unit(2)
w2 = Unit(3)
w3 = add(sin(w1), mul(w2, w1)) 
w3.backward_pass()
print(w1)
print(w2)

# sin(x)/cos(x) = tan(x)

x = Unit(30)
x1 = sin(x)
x2 = cos(x)
x3 = div(x1,x2)
x3.backward_pass()
print(x)

# z = w.x + b

w = Unit([2, 2, 2])
x = Unit([1, 1, 1])
b =Unit([1,1,1])
z = sum(add(dot(w, x), b))
z.backward_pass()
print(w)
print(x)
print(b)
