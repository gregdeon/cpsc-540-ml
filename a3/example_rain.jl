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
@show theta

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

# Q1.3: fit Markov model
# Fit parameters: probability of rain (X = 1)
# p1: p(rain) for day 1 of the month
p1 = sum(Xtrain[:, 1] .== 1) / (nTrain)
@show p1

# pt_0: p(rain | yesterday was sunny)
# pt_1: p(rain | yesterday was rainy)
rain_0 = 0
total_0 = 0
rain_1 = 0
total_1 = 0
for i in 1:nTrain
	for j in 2:d
		if Xtrain[i, j-1] == 0 
			# Yesterday was sunny
			global total_0 += 1
			if Xtrain[i, j] == 1
				global rain_0 += 1
			end
		else 
			# Yesterday was rainy
			global total_1 += 1
			if Xtrain[i, j] == 1
				global rain_1 += 1
			end
		end
	end
end

pt_0 = rain_0 / total_0 
pt_1 = rain_1 / total_1

@show pt_0
@show pt_1

# Measure validation set NLL
function measureNLL(X_NLL, p1, pt_0, pt_1)
	NLL = 0
	for i in 1:nValid
	    for j in 1:d
	    	if j == 1
    			# First day of the month
	    		theta = p1
			elseif X_NLL[i, j-1] == 0
    			# Yesterday was sunny
				theta = pt_0
			else
    			# Yesterday was rainy
				theta = pt_1
			end

	        if X_NLL[i,j] == 1
	            NLL -= log(theta)
	        else
	            NLL -= log(1-theta)
	        end
	    end
	end	
	return NLL
end

@show measureNLL(Xtrain, p1, pt_0, pt_1)
@show measureNLL(Xvalid, p1, pt_0, pt_1)


