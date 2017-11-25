# Load data
dataTable = readcsv("animals.csv")
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)
include("PCA.jl")
model = PCA(X, 2)
Z = model.compress(X)

# Plot matrix as image
using PyPlot
figure("PCA")
clf()

plot(Z[:,1], Z[:,2], ".")

for i in 1:n
    annotate(dataTable[i+1, 1], xy = [Z[i,1], Z[i,2]], xycoords = "data")
end
