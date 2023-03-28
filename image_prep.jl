using CSV
using FileIO
using Images
using DataFrames
using Flux
using QuartzImageIO

df = CSV.read("labeled_images.txt", DataFrame, header=true)

url = df.Images[1]
#print(url)
findlast("png", url)
#print(url[1:113])
file_path = download(url[1:113], "image.png", headers=Dict("User-Agent"=>"Mozilla/5.0"))
image1 = load(url[1:113])
print(image1)

for image in df.Images 
    image = load(image)
    image = imresize(image, (224, 224))
    image = channelview(image)
    image = Flux.preprocess(image)
end

print(df)