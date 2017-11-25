using(PyPlot)

function clustering2Dplot(X,y,W=[])
	(n,d) = size(X)

	k = length(unique(y))

	# Pick some symbols and colors for the clusters
	symbols = ["s","o","v","^","x","+","*","d","<",">","p"]
	colours = [(0,1,0)
			(1,0,0)
			(0,0,1)
			(1,0,1)
			(1,1,0)
			(0,1,1)
			(.1,.1,.1)
			(1,.5,0)
			(0,.5,0)
			(.5,.5,.5)
			(.5,.25,0)
			(.5,0,.5)
			(0,.5,1)]

	# Plot the points and the means
	clf()

	# Plot the points coloured by cluster
	for c in 1:k
		colour = (.75colours[c][1],.75colours[c][2],.75colours[c][3])
		plot(X[y.==c,1],X[y.==c,2],"o",marker=symbols[c],color=colour,markersize=5)

		if !isempty(W)			plot(W[c,1],W[c,2],marker=symbols[c],color=colours[c],markersize=12)
		end
	end

	# Plot the outliers in black
	if any(y.==0)
		plot(X[y.==0,1],X[y.==0,2],"o",color="black",markersize=5)
	end
end