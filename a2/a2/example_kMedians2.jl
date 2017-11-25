# Load data
using JLD
include("kMedians.jl")
X = load("clusterData2.jld","X")

# K-medians clustering



a= []
for k in 1:10
   minModel = GenericModel
   min = Inf
   for i in 1:50
      model = kMedians(X,k,doPlot = false)
      y = model.predict(X)
      temp = kMediansError(X,y,model.W)
      if min > temp
         min = temp
         minModel = model
      end
   end
   push!(a,min)
end



using PyPlot
figure()
title("kMedians the minimum error as k from 1 to 10 (clusterData2)")

plot(1:10,a,"b")
