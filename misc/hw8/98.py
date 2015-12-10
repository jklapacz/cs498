import numpy as np
a = list()
for i in range(983):
	a.append(1)
for i in range(1026):
	a.append(0)
print(a)
x = 0
y = 0
for i in range(len(a)):
	if(a[i] == 1):
		x += 1
	if(a[i] == 0):
		y += 1
print(x)
print(y)
mean = np.mean(a)
print(mean)
sd = np.std(a)
print(sd)