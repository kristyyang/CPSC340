# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

# Maximum depth we will plot
maxDepth = 30
t0 = time()
include("decisionTree.jl")
for depth in 1:maxDepth
	model = decisionTree(X,y,depth)
	yhat = model.predict(X)
	trainError = sum(yhat .!= y)/n
	@printf("Training error with depth-%d accuracy-based decision tree: %.2f\n",depth,trainError)
end

	@printf("Training time with accuracy-based decision tree: %.2f\n",time()-t0)

@printf("Now let's try infogain instead of accuracy...\n")

t1=time()
include("decisionTree_infoGain.jl")
for depth in 1:maxDepth
	model = decisionTree_infoGain(X,y,depth)

	yhat = model.predict(X)
	trainError = sum(yhat .!= y)/n
	@printf("Training error with depth-%d infogain-based decision tree: %.2f\n",depth,trainError)
end
	@printf("Training time with accuracy-based decision tree: %.2f\n",time()-t1)
