include("misc.jl")
include("clustering2Dplot.jl")

function dbCluster(X,radius,minPts;doPlot=false)
# Density-based clustering

	(n,d) = size(X)

	# The cluster of each object (0 means no cluster)
	y = zeros(Int64,n)

	# A vector to keep track of the examples we've "expanded"
	visited = fill(false,n)

	# Initially we have 0 clusters
	k = 1

	# Compute distances between all points
	D = sqrt.(max.(distancesSquared(X,X),0))

	# Set the distance from point to itself to be infinity
	# (we don't want the point itself as a neighbour)
	for i in 1:n
		D[i,i] = Inf
	end

	for i in 1:n
		if !visited[i]
			# We only need to consder example we haven't visited
			visited[i] = true

			# Find all the "close" points
			neighbours = find(D[:,i] .<= radius)

			# Test if this is a "core" point
			if length(neighbours) >= minPts

				# If it's a "core" point, we have a new cluster
				k += 1

				# "Expand" the cluster to find the rest of the cluster
				expandCluster!(X,i,neighbours,k,radius,minPts,D,visited,y,doPlot)
			end
		end
	end
	return y
end

function expandCluster!(X,i,neighbourStack,k,radius,minPts,D,visited,y,doPlot)
	# We use "neighbourStack" as a stack containing known
	# neighbours that may not yet have been expanded

	y[i] = k
	while !isempty(neighbourStack)
		n = pop!(neighbourStack)
		y[n] = k # Overwrites labels of boundary points

		if !visited[n]
			visited[n] = true
			neighbours2 = find(D[:,n] .<= radius)

			# Check if this point is a core point
			if length(neighbours2) >= minPts

				# If it's a core point, add neighbours to stack
				for n2 in neighbours2
					if !visited[n2]
						push!(neighbourStack,n2)
					end
				end
			end
		end

	end
end
