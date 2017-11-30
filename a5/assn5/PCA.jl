include("misc.jl")
include("findMin.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    # size(mu) = (d,1)
    mu = mean(X,1)
    #subtract rows by mu n times
    X -= repmat(mu,n,1)

    # x = u * diagm(s) *v'
    (U,S,V) = svd(X)
    W = V[:,1:k]'

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repmat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repmat(mu,t,1)
end


function PCA_gradient(X,k,sigma)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,1)
    X -= repmat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X

    f = (1/2)*sum(sqrt.(R.^2 + sigma))

    funObjZ(z) = pcaObjZ(z,X,W,sigma)
    funObjW(w) = pcaObjW(w,X,Z,sigma)

    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X

        f = (1/2)*sum(sqrt.(R.^2 + sigma))

        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu,sigma)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescent(Xhat,W,mu,sigma)
    (t,d) = size(Xhat)
    k = size(W,1)
    Xcentered = Xhat - repmat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZ(z,Xcentered,W,sigma)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZ(z,X,W,sigma)
    # Rezie vector of parameters into matrix
    (n,d) = size(X)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X

    f = (1/2)sum(sqrt.(R.^2 + sigma))

    # Comptue derivative with respect to each residual
    dR = (1/2)*R ./ sqrt.(R.^2 + sigma)

    # Multiply by W' to get elements of gradient
    G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjW(w,X,Z,sigma)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X


    f = (1/2)sum(sqrt.(R.^2 + sigma))

    # Comptue derivative with respect to each residual
    dR = (1/2) *R ./ sqrt.(R.^2 + sigma)

    # Multiply by Z' to get elements of gradient
    G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end


function robustPCA(X,k,sigma)
    return PCA_gradient(X,k,sigma)
end
