println("Adding packages...\n")

using HTTP, Images, Flux, Plots, Dates, LinearAlgebra, Random, Statistics, CSV, DataFrames, FixedPointNumbers, ColorTypes, Base.Threads, ImageContrastAdjustment, JLD2
using Flux: crossentropy, onecold, onehotbatch, train!, params, ADAM, @epochs

function download_image(url::String)
    response = HTTP.get(url)
    img_data = response.body
    img = Images.load(IOBuffer(img_data))
    return img
end

function resize_image(img, width::Int, height::Int)
    resized_img = imresize(img, (width, height))
    return resized_img
end

function increase_contrast(image::AbstractArray, contrast)
    alg = Equalization(nbins=contrast)
    return adjust_histogram!(image, alg)
end

function download_and_resize(url, image_size, contrast)
    image = download_image(url)
    image = resize_image(image, image_size, image_size)
    image = increase_contrast(image, contrast)
    return image
end

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

function load_data(image_size::Int64, contrast::Int64)
    print("Reading dataframe...\n")
    df = CSV.read("flux/labeled_images.csv", DataFrame)

    #get urls and labels
    urls = df[:, 1]
    labels = df[:, 2]

    #shuffle data
    indices = shuffle(1:length(urls))
    shuffled_urls = urls[indices]
    shuffled_labels = labels[indices]

    #size to split
    size_train = round(Int, length(shuffled_labels) * 0.80)
    size_test = round(Int, length(shuffled_labels) * 0.20)

    train_urls = shuffled_urls[1:size_train]
    train_labels = shuffled_labels[1:size_train]

    test_urls = shuffled_urls[size_train+1:size_train+size_test]
    test_labels = shuffled_labels[size_train+1:size_train+size_test]

    threads = nthreads()

    #do the same for test data
    n = length(test_urls)
    image_data_test = Vector{Any}(undef, n)
    counter = Atomic{Int64}(0)

    println("Starting $threads threads for test downloads\n")

    @threads for i in 1:n
        url = test_urls[i]
        image_data_test[i] = download_and_resize(url, image_size, contrast)
        cnt = atomic_add!(counter, 1)
        println("Downloaded $cnt images")
    end

    #convert images to float32 and make tensor for training data
    float32_image_data_test = [convert(Array{Float32}, float.(image)) for image in image_data_test]
    image_height, image_width = size(float32_image_data_test[1])
    num_images = length(image_data_test)
    X_test_raw = Array{Float32,3}(undef, image_height, image_width, num_images)

    @threads for i in 1:length(image_data_test)
        img = image_data_test[i]
        X_test_raw[:, :, i] = Float32.(img)
    end

    #get training data
    n = length(train_urls)
    image_data_train = Vector{Any}(undef, n)
    counter = Atomic{Int64}(0)

    println("Starting $threads threads for training downloads\n")

    @threads for i in 1:n
        url = train_urls[i]
        image_data_train[i] = download_and_resize(url, image_size, contrast)
        cnt = atomic_add!(counter, 1)
        println("Downloaded $cnt images")
    end

    #convert images to float32 and make tensor for training data
    float32_image_data_train = [convert(Array{Float32}, float.(image)) for image in image_data_train]
    image_height, image_width = size(float32_image_data_train[1])
    num_images = length(image_data_train)
    X_train_raw = Array{Float32,3}(undef, image_height, image_width, num_images)

    @threads for i in 1:length(image_data_train)
        img = image_data_train[i]
        X_train_raw[:, :, i] = Float32.(img)
    end

    # Reshape data to have channel dimension (C, H, W, N)
    X_train = reshape(X_train_raw, 1, image_height, image_width, num_images)
    X_test = reshape(X_test_raw, 1, image_height, image_width, size_test)

    #get labels into a one-hot encoded vector
    label_names = [" good", " bad"]
    y_train = onehotbatch(train_labels, label_names)
    y_test = onehotbatch(test_labels, label_names)

    return X_train, X_test, y_train, y_test
end

function load_test_data(image_size::Int64, contrast::Int64, num_test_images::Int64)
    print("Reading dataframe...\n")
    df = CSV.read("flux/labeled_images.csv", DataFrame)

    #get urls and labels
    urls = df[:, 1]
    labels = df[:, 2]

    #shuffle data
    indices = shuffle(1:length(urls))
    shuffled_urls = urls[indices]
    shuffled_labels = labels[indices]

    test_urls = shuffled_urls[1:num_test_images]
    test_labels = shuffled_labels[1:num_test_images]

    threads = nthreads()

    #do the same for test data
    n = length(test_urls)
    image_data_test = Vector{Any}(undef, n)
    counter = Atomic{Int64}(0)

    println("Starting $threads threads for test downloads\n")

    @threads for i in 1:n
        url = test_urls[i]
        image_data_test[i] = download_and_resize(url, image_size, contrast)
        cnt = atomic_add!(counter, 1)
        println("Downloaded $cnt images")
    end

    #convert images to float32 and make tensor for test data
    float32_image_data_test = [convert(Array{Float32}, float.(image)) for image in image_data_test]
    image_height, image_width = size(float32_image_data_test[1])
    num_images = length(image_data_test)
    X_test_raw = Array{Float32,3}(undef, image_height, image_width, num_images)

    @threads for i in 1:length(image_data_test)
        img = image_data_test[i]
        X_test_raw[:, :, i] = Float32.(img)
    end

    # Reshape data to have channel dimension (C, H, W, N)
    X_test = reshape(X_test_raw, 1, image_height, image_width, num_images)

    #get labels into a one-hot encoded vector
    label_names = [" good", " bad"]
    y_test = onehotbatch(test_labels, label_names)

    return X_test, y_test
end
