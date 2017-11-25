include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)

  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin

	#store distance between every testData and TrainData

	yhat = []
	for i in 1:t
		nearbyPoints = []
		for j in 1:n
			temp = 0

			for m in 1:d
				temp += (Xhat[i,m] - X[j,m]).^2
			end
			push!(nearbyPoints,sqrt(temp))
		end
		A = sortperm(nearbyPoints)
		push!(yhat ,mode(y[A[1:k]]))
	end

  return yhat
end



function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end

function cknn(X,y,k)
	# Implementation of condensed k-nearest neighbour classifier
	(n,d) = size(X)
	Xcond = X[1,:]'
	ycond = [y[1]]
	for i in 2:n
    	yhat = knn_predict(X[i,:]',Xcond,ycond,k)
    	if y[i] != yhat[1]
			Xcond = vcat(Xcond,X[i,:]')
			push!(ycond,y[i])
		end
	end
println(size(Xcond))
println(size(ycond))

	predict(Xhat) = knn_predict(Xhat,Xcond,ycond,k)
	return GenericModel(predict)
end
