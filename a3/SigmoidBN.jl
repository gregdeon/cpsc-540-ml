include("logReg.jl")
# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])
#X = X[:,:,1:100]
m = size(X,1)
@show m
n = size(X,3)
@show n

models = Array{SampleModel}(undef,m,m)
offset = 2
for i in 1:m
    for j in 1:m
        d = i*j
        Xsub = zeros(n,d)
        k = 1
        for ii in 1:i
            for jj in 1:j
                Xsub[:,k] = X[ii,jj,:]
                k+=1
            end
        end
        models[i,j] = logReg(Xsub[:,1:d-1],Xsub[:,d])
    end
end


# Fill-in some random test images
t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                d = i*j
                XtestSub = zeros(d)
                k = 1
                for ii in 1:i
                    for jj in 1:j
                        XtestSub[k] = I[ii,jj]
                        k+=1
                    end
                end
                I[i,j] = models[i,j].sample(XtestSub[1:d-1])
            end
        end
    end
    imshow(I)
end
