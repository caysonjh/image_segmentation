using CSV
using DataFrames
using HTTP
using Images

function prepare(response, iteration)
    image_data = response.body
    image = load(IOBuffer(image_data))
    resized_image = imresize(image, (224, 224))

    println("Finished with image at iteration: $iteration")

    return resized_image
end

df = CSV.read("labeled_images.txt", DataFrame, header=true)

# Create an array of tasks
tasks = [@async prepare(fetch(@async HTTP.get(url)), i) for (i, url) in enumerate(df.urls)]

# Wait for all tasks to complete and collect results
df.PixelVals = [fetch(t) for t in tasks]

CSV.write("prepared_images.csv", df[:, [:PixelVals, :Labels]])
