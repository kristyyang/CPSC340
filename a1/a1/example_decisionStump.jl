# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")

# Compute number of objects and number of features
(n,d) = size(X)

################################
### Majority Predictor Model ###
################################

# Fit majority predictor and compute error
include("majorityPredictor.jl")
model = majorityPredictor(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with majority predictor: %.2f\n",trainError);

################################
### Decision Stump Moodel ######
################################

# Fit decision stump classifier that uses equalities
#include("decisionStump.jl")
#model = decisionStumpEquality(X,y)


include("decisionStump.jl")
model = decisionStump(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with equality-rule decision stump: %.2f\n",trainError);

# Plot classifier
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model)
