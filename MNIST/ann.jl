using Flux, Images, MLDatasets, Plots, LinearAlgebra, Random, Statistics, DelimitedFiles
using Flux: crossentropy, onecold, onehotbatch, params, train!

Random.seed!(42)

# load data into memory
X_train_raw, y_train_raw = MLDatasets.MNIST.traindata(Float32)
X_test_raw, y_test_raw = MLDatasets.MNIST.testdata(Float32)

# view one of them
img = X_train_raw[:,:,1]
colorview(Gray, img') # img' is the transpose of img
println(y_train_raw[1])

# reshape our 3D tensor of training data (28x28x60000) to 2D tensor (784*60000)
X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

# one hot encode labels, reshapes our 60000 dim vector to 10*60000 dim 2D column-wise one hot tensor
y_train = onehotbatch(y_train_raw, 0:9)
y_test = onehotbatch(y_test_raw, 0:9)


# Define NN archiecture - Multilayer perceptron 
network = Chain(
                Dense(28*28, 32, relu),
                Dense(32,10),
                softmax
                )
# Define a loss
loss(x, y) = crossentropy(network(x), y)
# Optimizer to update weights (network parameters)
ws = params(network)
lr = 0.01
opt = ADAM(lr)


# train
loss_history = []
epochs = 500
 
for epoch in 1:epochs
    train!(loss, ws, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch | Training Loss = $train_loss")
end

x = 1:epochs
plot(x, loss_history)
savefig("loss.png") 
writedlm( "loss_history.csv",  loss_history, ',')