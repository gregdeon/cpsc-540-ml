
include("logReg.jl")
# Load X and y dataset
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])
m = size(X,1)
n = size(X,3)

models = Array{SampleModel}(undef,m,m)      # empty array
for i in 1:m
    for j in 1:m
        t = i*j                             # size of the region to consider
        X_tile = zeros(n,t)
        k = 1
        for p in 1:i
            for q in 1:j
                X_tile[:,k] = X[p,q,:]        # select all the samples at this location
                k+=1
            end
        end
        models[i,j] = logReg(X_tile[:,1:t-1],X_tile[:,t])
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
                t = i*j
                XtestSub = zeros(t)
                k = 1
                for p in 1:i
                    for q in 1:j
                        XtestSub[k] = I[p,q]
                        k+=1
                    end
                end
                I[i,j] = models[i,j].sample(XtestSub[1:t-1])
            end
        end
    end
    imshow(I)
end
