using LinearAlgebra
using Statistics
using Random
using Distributions
include("misc.jl") # Includes mode function and GenericModel typedef

function gaussianDensity(X)
	(n,d) = size(X)

	mu = (1/n)sum(X,dims=1)'
	Xc = X - repeat(mu',n)
	Sigma = (1/n)*(Xc'Xc)

	PDF(Xhat) = mvnpdf(Xhat,mu,Sigma)

	return DensityModel(PDF)
end

function mvnpdf(X,mu,Sigma)
	(n,d) = size(X)
	PDFs = zeros(n)

	logZ = (d/2)log(2pi) + (1/2)logdet(Sigma)
	SigmaInv = Sigma^-1
	for i in 1:n
		xc = X[i,:] - mu
		loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
		PDFs[i] = exp(loglik)
	end
	return PDFs
end

# em algorithm code
function EM(X,k,maxIter)
# initialize the parameters
(n,d) = size(X) # get the size of the data set
theta = ones(k) .* 1/k # weights for each k-clusters
mu = zeros(k,d) # means for the cluster centers
sigma = zeros(k,d,d) # each cluster has its own covariance matrix
R = zeros(n,k) # soft assignment - each example i belongs to every cluster partially

# for mu take any k values from X at random
# Choose random points to initialize means
perm = randperm(n)
for c = 1:k
	mu[c,:] = X[perm[c],:]
end

for i in 1:n
	R[i,:]= rand(Uniform(0,1),k)
	R[i,:] ./= sum(R[i,:])
end

for iter in 1:maxIter
	mu = zeros(k,d)
	#updates for mu's
	for i in 1:n
		for c in 1:k
			mu[c,:] += R[i,c].*X[i,:]
		end
	end

	#normalise
	for c in 1:k
		mu[c,:] ./= sum(R[:,c])
	end

	sigma = zeros(k,d,d)
	#update for covariance matrices
	for i in 1:n
		for c in 1:k
			xCentered = X[i,:]-mu[c,:]
			sigma[c,:,:] += R[i,c]*xCentered*xCentered'
		end
	end

	for c in 1:k
		sigma[c,:,:] ./= sum(R[:,c])
	end
	#update rule for responsibilities
	for i in 1:n
		for c in 1:k
			R[i,c] = theta[c] * mvnpdf(X[i,:]',mu[c,:],sigma[c,:,:])[1]
		end
		R[i,:] ./= sum(R[i,:])
	end
	#update rule for theta
	for c in 1:k
		theta[c] = sum(R[:,c])/n
	end
end
function pdf(Xhat)
	(n,d) = size(Xhat)
	pdfs = zeros(n)
	for i in 1:n
		for c in 1:k
			pdfs[i] += theta[c]*mvnpdf(Xhat[i,:]',mu[c,:],sigma[c,:,:])[1]
		end
	end
	return pdfs
end
 return DensityModel(pdf)
end
