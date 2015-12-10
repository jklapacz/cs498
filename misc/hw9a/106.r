# Part A
seedsdata <- read.table('seeds_dataset.txt', header = FALSE)
numseeds = seedsdata[,c(1:7)]
SEEDS = cbind(numseeds$V1, numseeds$V2, numseeds$V3, numseeds$V4, numseeds$V5, numseeds$V6, numseeds$V7)
COVSEEDS = cov(SEEDS)
EIGENWINE = eigen(COVSEEDS)
pca <- princomp(numseeds)
pcar <- prcomp(numseeds)
scaled <- scale(numseeds, pcar$center, pcar$scale) %*% pcar$rotation
xl <- seedsdata[,8]
x <- scaled[, 1:2]
biplot(x, pca$scores[1:2, ], xlabs = xl, ylabs=c('Component1', 'Component2'), main="Principal Component Projection", ylab="Scaled Component 2", xlab="Scaled Component 1")

# Part B
sorted = sort(EIGENWINE$values, decreasing = TRUE)
plot(sorted, type='b', main="Eigenvalues of Covmat(Seeds)", ylab="Eigenvalue", xlab="Component")
