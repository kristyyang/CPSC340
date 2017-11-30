include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end



function ISOMAP(X,k)
    (n,d) = size(X)

    temp = distancesSquared(X,X)
    temp = sqrt.(abs.(temp))

    weight = fill(Inf, n,n)

    for i in 1:n
        v = sortperm(temp[i,:])
        weight[i,v[1:k]] = temp[i,v[1:k]]
        weight[v[1:k],i] = temp[i,v[1:k]]
    end

    D = zeros(n,n)

    for i in 1:n
        for j in 1:n
            D[i,j] = dijkstra(weight,i,j)
        end
    end

    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function ISOMAP2(X,k)
    (n,d) = size(X)

    temp = distancesSquared(X,X)
    temp = sqrt.(abs.(temp))

    weight = fill(Inf,n,n)

    for i in 1:n
        v = sortperm(temp[i,:])
        weight[i,v[1:k]] = temp[i,v[1:k]]
        weight[v[1:k],i] = temp[i,v[1:k]]
    end

    D = zeros(n,n)

    for i in 1:n
        for j in 1:n
            D[i,j] = dijkstra(weight,i,j)
        end
    end

    a = findmax(filter(!isinf, D))

    D[isinf(D)] = a[1];

    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end
