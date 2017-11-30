# Load X variable
using JLD
X = load("highway.jld","X")
(n,d) = size(X)

# Fit PCA model
include("PCA.jl")
sigma = 0.0001
k = 5
model = robustPCA(X,k,sigma)

Z = model.compress(X)
Xhat = model.expand(Z)

using PyPlot
nFrames = 50 # Number of frames to show
pause = 0.1 # How long to pause on each frame
for i in 1:nFrames
	original = reshape(X[i,:],64,64)
	reconstr = reshape(Xhat[i,:],64,64)
	diff = 255*(abs.(original-reconstr) .> 10)
	figure(1)
	clf()
	imshow([original reconstr diff],cmap="gray")
	sleep(pause)
end