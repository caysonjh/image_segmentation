include("data_prep.jl")
include("beep.jl")
using Flux.Optimise, JLD2
using Flux: flatten

image_size = 224
contrast = 50

function get_data(image_size::Int64, contrast::Int64)
  file_path = "flux/image_datasets/$image_size-px-$contrast.jld2"
  if isfile(file_path)
    println("Loading a dataset of images sized $image_size x $image_size with a contrast of $contrast")
    @load file_path X_train X_test y_train y_test
    return X_train, X_test, y_train, y_test
  else
    X_train, X_test, y_train, y_test = load_data(image_size, contrast)
    @save file_path X_train X_test y_train y_test
    println("Dataset images sized $image_size x $image_size with a contrast of $contrast successfully created")
    return X_train, X_test, y_train, y_test
  end
end

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
    Dense(prod(out_conv_size), 120, relu),
    Dense(120, 84, relu),
    Dense(84, nclasses),
    softmax
  )
end

model = build_cnn()

#define loss function, params, optimizer, and live plot
loss(x, y) = crossentropy(model(x), y)
learning_rate = 0.000001
ps = params(model)
opt = ADAM(learning_rate)
loss_plot = Plots.plot([], [], xlabel="Epochs", ylabel="Loss", title="Live Loss Plot", legend=false, color=:blue, linewidth=2)
display(loss_plot)

println("Let the training begin!\n")

#train model
loss_history = []
epochs = 2
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

@save "flux/models/model_$start_time.jld2" model

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