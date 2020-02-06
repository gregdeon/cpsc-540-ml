include("misc.jl") # Includes mode function and GenericModel typedef
include("studentT.jl")

function tda(X,y)
  (n,d) = size(X)
  k = maximum(y)

  submodels = Array{DensityModel}(undef, k)
  theta = zeros(k)

  for yi in 1:k
    i_class = (y .== yi)
    X_yi = X[i_class, :]
    submodels[yi] = studentT(X_yi)
    theta[yi] = sum(i_class) / n
  end

  function predict(X_hat)
    (t, d) = size(X_hat)
    y_hat = zeros(Int, t)
    for i in 1:t
      PDFs_i = zeros(k)
      for j in 1:k
        PDFs_i[j] = theta[j] * submodels[j].pdf(X_hat[i, :]')[1]
      end
      y_hat[i] = argmax(PDFs_i)
    end
    return y_hat
  end
  return GenericModel(predict)
end


