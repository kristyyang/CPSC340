using PyCall
using PyPlot

function plot2Dclassifier(X,y,model)

	increment = 100

	figure()
	plot(X[y.==1,1],X[y.==1,2],"b+")
	plot(X[y.==2,1],X[y.==2,2],"ro")

	(xmin,xmax) = xlim()
	xDomain = linspace(xmin,xmax,increment)
	(ymin,ymax) = ylim()
	yDomain = linspace(ymin,ymax,increment)

	xValues = repmat(xDomain,1,length(xDomain))
	yValues = repmat(yDomain',length(yDomain),1)

	z = model.predict([xValues[:] yValues[:]])

	@assert(length(z) == length(xValues),"Size of model function's output is wrong");

	zValues = reshape(z,size(xValues))

	if all(zValues[:] == 1)
    		cm = [(0,0,.5)];
	elseif all(zValues[:] == 2)
    		cm = [(.5,0,0)];
	else
    		cm = [(0,0,.5);(.5,0,0)];
	end
	@pyimport matplotlib.colors as matcolors
	cmap = matcolors.ListedColormap(cm,"A")

	contourf(xValues,yValues,zValues,cmap=cmap)
end
