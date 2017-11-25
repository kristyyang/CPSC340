# Load data
using JLD
fileName = "vowel.jld"
X = load(fileName,"X")
y = load(fileName,"y")
Xtest = load(fileName,"Xtest")
ytest = load(fileName,"ytest")

# Fit a decision tree classifier
include("decisionTree.jl")
depth = Inf
model = decisionTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d decision tree: %.3f\n",depth,testError)

# Fit a random tree classifier
include("decisionTree.jl")
depth = 5
model = randomTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random tree: %.3f\n",depth,testError)


# Fit a random tree classifier
include("decisionTree.jl")
nTree = 50
depth = Inf
model = randomForest(X,y,depth,nTree)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random forest: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random forest: %.3f\n",depth,testError)
