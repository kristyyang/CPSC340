# Load X and y variable
using JLD
using PyCall

include("decisionTree_infoGain.jl")
include("decisionTree.jl")
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

a = size(X)[1];
#b = convert(Interger,floor(a/2));

# Train a depth-2 decision tree
maxDepth = 15

Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")
#Xtrain = X[201:end,:]
#ytrain= y[201:end,:]

#Xtest = X[1:200,:]
#ytest = y[1:200,:]

trainError = []
testError = []
for depth in 1:maxDepth
  model = decisionTree_infoGain(X,y,depth)
  #Evaluate the trianing error
  yhat = model.predict(X)
  q = size(X,1)
  push!(trainError, sum(yhat .!= y)/q)
  #@printf("Train error with depth-%d decision tree: %.3f\n",depth,trainError[depth])

  # Evaluate the test error
  t = size(Xtest,1)
  yhat = model.predict(Xtest)
  push!(testError, sum(yhat .!= ytest)/t)
  #@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError[depth])
end


@pyimport numpy
@pyimport pylab
pylab.plot(1:maxDepth, trainError,label = "trainError"; color="red", linewidth=2.0, linestyle="--")
pylab.plot(1:maxDepth, testError,label = "testError"; color="green", linewidth=2.0, linestyle="-")
pylab.legend()
pylab.title("trainTest")
pylab.show()
