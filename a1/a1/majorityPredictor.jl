include("misc.jl") # Includes "mode" function

function majorityPredictor(X,y)
	# Model that always predicts most common y

	# Compute most common y value
	y_mode = mode(y)

	# The model is a function that 
	# says "y_mode" for all test examples
	predict(Xhat) = fill(y_mode,size(Xhat,1))

	return GenericModel(predict)
end

