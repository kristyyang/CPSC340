# Load data
using JLD
include("kMedians.jl")
X = load("clusterData2.jld","X")

# K-means clustering

minModel = GenericModel

min = Inf

Min = []

for i in 1:50
   k = 4
   model = kMedians(X,k,doPlot = false)
   y = model.predict(X)
   temp = kMediansError(X,y,model.W)
   if min > temp
      min = temp
      minModel = model
      Min = y
   end
   #println(min)
end
#println(min)

y= minModel.predict(X)


include("clustering2Dplot.jl")
clustering2Dplot(X,Min,minModel.W)
