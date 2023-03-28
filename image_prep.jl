using CSV
using FileIO
using Images
using DataFrames
using Flux
using QuartzImageIO
using PyCall
using HTTP
using ImageView

function prepare(url)
    local response = HTTP.get(url)
    local image_data = response.body
    local image = load(IOBuffer(image_data))
    local resized_image = imresize(image, (224, 224))

    println("Finished with image")

    return resized_image
end

df = CSV.read("labeled_images.txt", DataFrame, header=true)

df.PixelVals = [prepare(url) for url in df.Images]

CSV.write("prepared_images.csv", df[:, [:PixelVals, :Labels]])