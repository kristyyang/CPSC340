include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBias(X,y)
	(n,) = size(X)

	M = ones(n)

	Xn = hcat(M,X)

	# Find regression weights minimizing squared error
	w = (Xn'*Xn)\(Xn'*y)

	# Make linear prediction function
	function predict(Xhat)
		(n,)= size(Xhat)

		N = ones(n)

		Xhats = hcat(N,Xhat)

		return Xhats*w
	end

	return GenericModel(predict)
end


function leastSquaresBasis(X,y,p)
	(u,) = size(X)

	Z = ones(u,p)

	for i in 1:p
		#for j in 1:u
			Z[:,i] = X.^(i)
		#end
	end

	M = ones(u)
	newz = hcat(M,Z)

	w = (newz'*newz)\(newz'*y)
	# Make linear prediction function
	function predict(Xhat)
		(u,) = size(Xhat)

		Z = ones(u,p)
		for i in 1:p
			#for j in 1:u

				Z[:,i] = Xhat.^(i)
			#end
		end

		N = ones(u)

		newZh = hcat(N,Z)

		return newZh*w
	end

	return GenericModel(predict)
end


function weightedLeastSquares(X,y,v)

	w = (X'*v*X)\(X'*v*y)
	predict(Xhat) = Xhat*w

	return GenericModel(predict)
end
