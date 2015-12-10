import numpy as np
a = np.array([21, 23, 27,19, 17, 18, 20, 15, 17, 22])
mean = np.mean(a)
sd = np.std(a)
b = a - mean

print(b)
b = b * b
meanb = np.mean(b)
sqb = np.sqrt(meanb)
print(sqb)
print(meanb)
print("Mean = " + str(mean))
print("Standard Deviation = " + str(sd))
