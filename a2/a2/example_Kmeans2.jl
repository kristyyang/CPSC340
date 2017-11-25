# Load data
using JLD
include("kMeans.jl")
X = load("clusterData2.jld","X")

# K-means clustering



a= []
for k in 1:10
   minModel = GenericModel
   min = Inf
   for i in 1:50
      model = kMeans(X,k,doPlot = false)
      y = model.predict(X)
      temp = KMeansError(X,y,model.W)
      if min > temp
         min = temp
         minModel = model
      end
   end
   push!(a,min)
end


using PyPlot
figure()
title("k-means the minimum error as k from 1 to 10 (clusterData)")

plot(1:10,a,"b")
