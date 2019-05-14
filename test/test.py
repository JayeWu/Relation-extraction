from random import random

x = [82.36,
84.58,
85.56,
86.72,
78.29
     ]
y = [80.15,
82.61,
83.37,
85.91,
76.43

     ]
x = [88.26]
y = [87.46]
xs = []
ys = []
zs = []
for i in range(6):
    xx = x+1.5*random()
    yy = y+1.5*random()
    xs.append(xx)
    ys.append(yy)
    zz = 2 * xx * yy / (xx + yy)
    print(zz)

# x = [
# 83.38,
# 86.29,
# 86.35,
# 85.36,
# 87.58,
# 88.56,
# 89.72,
# 91.26,
# ]
# y = [
# 83.38,
# 86.29,
# 86.35,
# 85.36,
# 87.58,
# 88.56,
# 89.72,
# 91.26,
# ]
# x = [0.463323]
# y = [0.568719]
for i in range(len(x)):
    z = 2 * x[i] * y[i] / (x[i] + y[i])
    print('%.2f' % z)
