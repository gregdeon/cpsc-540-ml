include("tabular.jl")
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
for i in 15:m
    for j in 1:m
        i1 = max(i-2,1)
        j1 = max(j-2,1)
        d = (i-i1+1)*(j-j1+1)
        Xsub = zeros(n,d)
        k = 1
        for ii in i1:i
            for jj in j1:j
                Xsub[:,k] = X[ii,jj,:]
                k +=1
            end
        end
        models[i,j] = tabular(Xsub[:,1:d-1],Xsub[:,d])
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
                i1 = max(i-2,1)
                j1 = max(j-2,1)
                d = (i-i1+1)*(j-j1+1)
                XtestSub = zeros(d)
                k =1
                for ii in i1:i
                    for jj in j1:j
                        XtestSub[k] = I[ii,jj]
                        k+=1
                    end
                end
                #print(XtestSub)
                I[i,j] = models[i,j].sample(XtestSub[1:d-1])
            end
        end
    end
    imshow(I)
end
