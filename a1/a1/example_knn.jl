# Load X and y variable
using JLD


#X = load("citiesBig2.jld","X")
#y = load("citiesBig2.jld","y")
#Xtest = load("citiesBig2.jld","Xtest")
#ytest = load("citiesBig2.jld","ytest")


X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")

# Fit a KNN classifier
k = 4
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)
