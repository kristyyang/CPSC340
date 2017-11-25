include("misc.jl")
include("clustering2Dplot.jl")

type PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
end

function kMeans(X,k;doPlot=false)
# K-means clustering

(n,d) = size(X)

# Choos random points to initialize means
W = zeros(k,d)
perm = randperm(n)
for c = 1:k
	W[c,:] = X[perm[c],:]
end

# Initialize cluster assignment vector
y = zeros(n)
changes = n

while changes != 0

	# Compute (squared) Euclidean distance between each point and each mean
	D = distancesSquared(X,W)

	# Assign each data point to closest mean (track number of changes labels)
	changes = 0
	for i in 1:n
		(~,y_new) = findmin(D[i,:])
		changes += (y_new != y[i])
		y[i] = y_new
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end


	# Find mean of each cluster
	for c in 1:k
		W[c,:] = mean(X[y.==c,:],1)
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	m = KMeansError(X,y,W)
	@printf("Running k-means, changes = %d\n",changes)
	@printf("Running k-means, kMeansError is = %.3f\n", m)
end

function predict(Xhat)
	(t,d) = size(Xhat)

	D = distancesSquared(Xhat,W)

	yhat = zeros(Int64,t)
	for i in 1:t
		(~,yhat[i]) = findmin(D[i,:])
	end
	return yhat
end

return PartitionModel(predict,y,W)
end


function KMeansError(X,y,W)
	(k,m) = size(X)

	n =0
	for i in 1:k
		for j in 1:m
			n += (X[i,j] - W[Int(y[i]),j]).^2
		end
	end
		return n
end
