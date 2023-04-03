using JLD2, Flux, Statistics
using Flux: onecold

@load "flux/image_datasets/224-px-50.jld2" X_train X_test y_train y_test

@load "flux/image_datasets/1stCNN.jld2" model

image_size = 224

X_test = reshape(X_test, image_size, image_size, 1, :)

single = X_test[:, :, 1, 1] #get the first image

single = reshape(test, image_size, image_size, 1, :)


y_hat_raw = model(single)

y_hat = onecold(y_hat_raw) .- 1

y = onecold(y_test) .- 1

accuracy = mean(y_hat .== y)

good = 0
bad = 0
for i in 1:length(y_hat)
  if y_hat[i] == 1
    good += 1
  else
    bad += 1
  end
end

good
bad