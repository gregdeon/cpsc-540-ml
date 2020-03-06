include("tabular.jl")
# Load X and y dataset
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

models = Array{SampleModel}(undef,m,m)  # empty array
for i in 15:m                           # since half we already know
    for j in 1:m
        rows_window = max(i-2,1)        # number of rows in the submatrx
        columns_window = max(j-2,1)     # number of cols in the submatrix
        t = (i-rows_window+1)*(j-columns_window+1) # number of entries
        X_tile = zeros(n,t)
        k = 1
        for p in rows_window:i
            for q in columns_window:j
                X_tile[:,k] = X[p,q,:]      # select all the samples at this location
                k +=1
            end
        end
        models[i,j] = tabular(X_tile[:,1:t-1],X_tile[:,t])
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
                rows_window = max(i-2,1)
                columns_window = max(j-2,1)
                t = (i-rows_window+1)*(j-columns_window+1)
                XtestSub = zeros(t)
                k =1
                for p in rows_window:i
                    for q in columns_window:j
                        XtestSub[k] = I[p,q]
                        k+=1
                    end
                end
                #print(XtestSub)
                I[i,j] = models[i,j].sample(XtestSub[1:t-1])
            end
        end
    end
    imshow(I)
end
