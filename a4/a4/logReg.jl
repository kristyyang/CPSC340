include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 + exp.(-yXw)))
	g = -X'*(y./(1+exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] = -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end

function softmaxClass(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	# Each binary objective has the same features but different lables
	funObj(w) = SoftObj(w,X,y)


	w = findMin(funObj,reshape(W,d*k),verbose=false, derivativeCheck =true)

    W= reshape(w,d,k)
	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat * W,2)

	return LinearModel(predict,W)
end

function SoftObj(w,X,y)
	(n,d) = size(X)
	k = maximum(y)


	sumlog0 = 0
	sumlog1 = 0
	f = 0
	W = reshape(w,d,k)

	for i in 1:n
		f += -(W[:,y[i]]'*X[i,:])
		for c in 1: k
			sumlog0 += exp.(W[:,c]'*X[i,:])
		end
		f += log(sumlog0)
		sumlog0  = 0
	end

	g0 = zeros(d,k)
	g1 = zeros(n,d)

	for c in 1: k
		for i in 1: n
			sumlog1 = sum(exp.(W'*X[i,:]))
			g1[i,:] = -X[i,:]*(y[i]==c) + (exp.(W[:,c]'* X[i,:])) * X[i,:]/sumlog1
		end
		g0[:,c] = sum(g1,1)'
	end

	g0= reshape(g0, d*k)

	return (f,g0)
end
