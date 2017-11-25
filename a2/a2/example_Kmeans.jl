# Load data
using JLD
include("kMeans.jl")
X = load("clusterData2.jld","X")

# K-means clustering

minModel = GenericModel

min = Inf

for i in 1:50
   k = 4
   model = kMeans(X,k,doPlot = false)
   y = model.predict(X)
   temp = KMeansError(X,y,model.W)
   if min > temp
      min = temp
      minModel = model
   end
   println(min)
end
println(min)

y= minModel.predict(X)


include("clustering2Dplot.jl")
clustering2Dplot(X,y,minModel.W)
