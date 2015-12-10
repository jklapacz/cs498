# Part A
winedat<-read.csv('wine.data', header=FALSE);
library('lattice');
numwine = winedat[,c(2:14)];
eigenwine = eigen(numwine);
WINE = cbind(numwine$V2,numwine$V3,numwine$V4,numwine$V5,numwine$V6,numwine$V7,numwine$V8,numwine$V9,numwine$V10,numwine$V11,numwine$V12,numwine$V13,numwine$V14);
COV = cov(WINE);
EIGEN = eigen(COV);
sorted = sort(EIGEN$values, decreasing = TRUE);
plot(sorted, type='b', main="Eigenvalues of Covmat(Wine)", ylab="Eigenvalue", xlab="Component");

# Part B
d<- EIGEN$vectors[c(1:3),c(1:13)];
plot(d, type='p');
lines(d, type='p');
lines(d, type='h');
plot(d[1, 1:13], type='p', col="red");
lines(d[2, 1:13], type='p', col="blue");
lines(d[3, 1:13], type='p', col="green");
lines(d[1, 1:13], type='h', col="red");
lines(d[2, 1:13], type='h', col="blue");
lines(d[3, 1:13], type='h', col="green");
title(main="Principal Components", sub="Red = comp1\n Blue = comp2\n Green = comp3", xlab="");
title(ylab="Value");

# Part C
pca <- princomp(numwine);
pcar = prcomp(numwine);
scaled <- scale(numwine, pcar$center, pcar$scale) %*% pcar$rotation;
xl = winedat[, 1];
x = scaled[, 1:2];
biplot(x, pca$scores[1:2, ], xlabs = xl, ylabs=c('Component1', 'Component2'), main="Principal Component Projection", ylab="Scaled Component 2", xlab="Scaled Component 1");

