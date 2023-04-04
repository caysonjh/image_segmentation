include("data_prep.jl")
include("beep.jl")
using Flux.Optimise, JLD2
using Flux: flatten

learning_rate = 0.000001
epochs = 50

image_size = 224
contrast = 10

download_and_resize("https://storage.labelbox.com/cl8adr7vlae2807xb1fnc5x7e%2Fdc79ed98-2639-fff2-352c-4dab1942540e-PginABK7_288.h5.png?Expires=1681179638094&KeyName=labelbox-assets-key-3&Signature=eaoyiYxVl3_c_3PVRj_U5qeleS0", image_size, contrast)

X_train, X_test, y_train, y_test = @beep get_data(image_size, contrast)

X_train = reshape(X_train, image_size, image_size, 1, :)
X_test = reshape(X_test, image_size, image_size, 1, :)

y_train
y_test

#define model
function build_cnn(; imgsize=(image_size, image_size, 1), nclasses=2)
  out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

  return Chain(
    Conv((5, 5), imgsize[end] => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(prod(out_conv_size), 64, relu),
    Dense(64, nclasses),
    softmax
  )
end

model = build_cnn()

#define loss function, params, optimizer, and live plot
loss(x, y) = crossentropy(model(x), y)
ps = params(model)
opt = ADAM(learning_rate)
loss_plot = Plots.plot([], [], xlabel="Epochs", ylabel="Loss", title="Live Loss Plot", legend=false, color=:blue, linewidth=2)
display(loss_plot)

println("Let the training begin!\n")

#train model
loss_history = []
start_time = now()
@beep for epoch in 1:epochs
  #train with optimized learning rate
  train!(loss, ps, [(X_train, y_train)], opt)
  train_loss = loss(X_train, y_train)

  #add loss value to the loss_history list
  push!(loss_history, train_loss)
  println("Epoch = $epoch : Training loss = $train_loss")

  #update plot and refresh display
  Plots.plot!(loss_plot, 1:epoch, loss_history, xlabel="Epochs", ylabel="Loss", title="Live Loss Plot", legend=false, color=:blue, linewidth=2)
  display(loss_plot)
end

@save "flux/models/model_$image_size-px-$contrast.jld2" model

end_time = now()
elapsed_time = end_time - start_time
elapsed_seconds_total = Dates.value(elapsed_time) ÷ 1000
minutes = elapsed_seconds_total ÷ 60
seconds = elapsed_seconds_total % 60
println("Training time: $minutes minutes and $seconds seconds")

#run the model on the test data
y_hat_raw = model(X_test)

#turn the probabilities into labels with onecold()
y_hat = onecold(y_hat_raw) .- 1

#compare the predicitons (y_hat) to the actual true labels (y_hat_raw)
y = onecold(y_test) .- 1

accuracy = mean(y_hat .== y)

#plot and save the loss funciton with respect to epochs
gr(size=(600, 600))
p_1_curve = Plots.plot(1:epochs, loss_history, xlabel="Epochs", ylabel="Loss", title="Test Accuracy: $accuracy | Time: $minutes:$seconds | LR: $learning_rate", legend=false, color=:blue, linewidth=2)

savetime = now()
savefig(p_1_curve, "flux/model_graphs/model_learning_curve_$savetime.png")