# Load X and y variable
using JLD

# Load initial probabilities and transition probabilities of Markov chain
data = load("rain.jld")
X = data["X"]
(n,d) = size(X)

# Split into a training and validation set
splitNdx = Int(ceil(n/2))
trainNdx = 1:splitNdx
validNdx = splitNdx+1:n
Xtrain = X[trainNdx,:]
Xvalid = X[validNdx,:]
nTrain = length(trainNdx)
nValid = length(validNdx)

# Fit a single Bernoulli to the entire dataset
theta = sum(Xtrain .== 1)/(nTrain*d)

# Measure test set NLL
NLL = 0
for i in 1:nValid
    for j in 1:d
        if Xvalid[i,j] == 1
            global NLL -= log(theta)
        else
            global NLL -= log(1-theta)
        end
    end
end
@show NLL
