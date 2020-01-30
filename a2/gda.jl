include("misc.jl") # Includes mode function and GenericModel typedef
include("gaussianDensity.jl")

function gda(X,y)
  (n,d) = size(X)
  k = maximum(y)

  submodels = Array{DensityModel}(undef, k)

  for yi in 1:k
    X_yi = X[y .== yi, :]
    submodels[yi] = gaussianDensity(X_yi)
  end

  function predict(X_hat)
    (t, d) = size(X_hat)
    y_hat = zeros(Int, t)
    for i in 1:t
      PDFs_i = zeros(k)
      for j in 1:k
        PDFs_i[j] = submodels[j].pdf(X_hat[i, :]')[1]
      end
      y_hat[i] = argmax(PDFs_i)
    end
    return y_hat
  end
  return GenericModel(predict)
end
