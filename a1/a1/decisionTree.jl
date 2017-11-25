include("decisionStump.jl")

function decisionTree(X,y,depth)
	# Fits a decision tree using greedy recursive splitting
	# (recursion to make the code simpler)

	(n,d) = size(X)

	# Learn a decision stump
	splitModel = decisionStump(X,y)

	if depth <= 1 || splitModel.baseSplit
		# Base cases where we stop splitting:
		# - this stump gets us to the max depth
		# - this stump doesn't split the data
		return splitModel
	else
		# Use the decision stump to split the data
		yes = splitModel.split(X)

		# Recusively fit a decision tree to each split
		yesModel = decisionTree(X[yes,:],y[yes],depth-1)
		noModel = decisionTree(X[.!yes,:],y[.!yes],depth-1)
		
		# Make a predict function
		function predict(Xhat)
			(t,d) = size(Xhat)
			yhat = zeros(t)

			yes = splitModel.split(Xhat)

			yhat[yes] = yesModel.predict(Xhat[yes,:])
			yhat[.!yes] = noModel.predict(Xhat[.!yes,:])
			return yhat
		end

		return GenericModel(predict)
	end
end

	
	