using LinearAlgebra
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
