# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)
#validStart = Int64(n/2+1) # Start of validation indices
#validEnd = Int64(n) # End of validation incides
#validNdx = perm[validStart:validEnd] # Indices of validation examples
#trainNdx = perm[setdiff(1:n,validNdx)] # Indices of training examples

stop = floor(Int,(n / 10))
println(stop)

validIndex = Matrix{Int64}(10, stop)
trainIndex = Matrix{Int64}(10,n-stop)

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []

for i in 1: 10
	validEnd = stop * i
	validStart = validEnd - stop + 1
	validIndex[i,:] = perm[validStart:validEnd]
	trainIndex[i,:] = perm[setdiff(1:n,validStart:validEnd)]
end


	for sigma in 2.0.^(-15:15)
		sumErr = 0
		for i in 1:10
			Xtrain = X[trainIndex[i,:],:]
			ytrain = y[trainIndex[i,:],:]
			Xvalid = X[validIndex[i,:],:]
			yvalid = y[validIndex[i,:],:]
			# Compute the error on the validation set
			model = leastSquaresRBF(Xtrain,ytrain,sigma,10^-12.0)
			yhat = model.predict(Xvalid)
			validError = sum((yhat - yvalid).^2)/(9*n/10)
			#@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)
			# Keep track of the lowest validation error
			sumErr += validError
		end

		avgErr = sumErr/10
		if avgErr < minErr
			minErr = avgErr
			bestSigma = sigma
		end
	end

#for i in 1:10
	# Train on the training set
	#model = leastSquaresRBF(Xtrain,ytrain,sigma,10^-12.0)
#	validEnd = stop * i
#	validStart = validEnd - stop + 1
#	if validStart == 1
#		Xvalid = X[Int(validStart):Int(validEnd),:]
#		Xtrain = X[Int(validEnd)+1:end,:]
#		ytrain = y[Int(validEnd)+1:end,:]
#		yvalid = y[Int(validStart):Int(validEnd),:]
#	elseif validEnd != n
#		Xvalid = X[Int(validStart):Int(validEnd),:]
#		Xtrain1 = X[1:Int(validStart)-1,:]
#		yvalid  = y[Int(validStart):Int(validEnd),:]
#		Xtrain2 = X[Int(validEnd)+1 :end,:]
#		Xtrain = vcat(Xtrain1,Xtrain2)
#		ytrain1 = y[1:Int(validStart)-1,:]
#		ytrain2 = y[1+Int(validEnd):end,:]
#		ytrain = vcat(ytrain1,ytrain2)
#	else
#		Xvalid = X[Int(validStart):Int(validEnd),:]
#		Xtrain = X[1:Int(validStart)-1,:]
#		yvalid = y[Int(validStart):Int(validEnd),:]
#		ytrain = y[1:Int(validStart)-1,:]
#	end



#	for sigma in 2.0.^(-15:15)
#	# Compute the error on the validation set
#	model = leastSquaresRBF(Xtrain,ytrain,sigma,10^-12.0)
##	yhat = model.predict(Xvalid)
#	validError = sum((yhat - yvalid).^2)/(n/2)
#	@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)

	# Keep track of the lowest validation error
#	if validError < minErr
#		minErr = validError
#		bestSigma = sigma
#	end
#end
#end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma,10^-12.0)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
