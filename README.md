# SCRATCH-DIFF : implementation of autodifferentiation in python

This repo does not contain any clever or fancy implementation of autodifferentiation. Its heavily inspired by older blogs and repos. The code is heavily simplified and not at all optimized XD. The primary goal is to learn and understand how it really works. 

```
import scratchdiff.Unit as sd
from scratchdiff.Unit import Unit

w1 = Unit(2)
w2 = Unit(3)
w3 = sd.add(sd.sin(w1), sd.mul(w2, w1)) 
w3.backward_pass()
print(w1.grad.data)
print(w2.grad.data)
```