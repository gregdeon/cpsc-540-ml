using JLD, Printf, Statistics

# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a KNN classifier
k = 1
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

# Q2.3.3: fit GDA classifer
include("gda.jl")
gda_model = gda(X, y)

# Evaluate training error
yhat = gda_model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with GDA: %.3f\n", trainError)

# Evaluate test error
yhat = gda_model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with GDA: %.3f\n", testError)

# Q2.3.4: fit TDA classifier
include("tda.jl")
tda_model = tda(X, y)

# Evaluate training error
yhat = tda_model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with TDA: %.3f\n", trainError)

# Evaluate test error
yhat = tda_model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with TDA: %.3f\n", testError)

