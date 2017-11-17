include("Arff.jl")
include("knn.jl")

using Plots

# push!(LOAD_PATH, ".")

importall Arff
importall KNN

train_arff = loadarff("data/MagicTelescopeTrain.arff")
test_arff = loadarff("data/MagicTelescopeTest.arff")

train_x = convert(Matrix{Float64}, train_arff.data[:, 1:end-1])
train_y = train_arff.data[:, end:end]
test_x = convert(Matrix{Float64}, test_arff.data[:, 1:end-1])
test_y = test_arff.data[:, end:end]

model = Knn(train_x, train_y)

# Graph for all odd values between k = 1 and k = 15
accuracy = Vector{Float64}(8)
for i = 1:8
    println(i)
    correct  = classify(model, test_x, (i*2) - 1) .== test_y[:, 1]
    accuracy[i] = mean(correct)
end

function plotacc()
    pyplot()
    plot(accuracy, xticks=(1:8, 1:2:15), label="Test Accuracy")
    title!("Accuracy on Magic Telescope Dataset with Varying k")
    xaxis!("k")
    yaxis!("Accuracy")
    ylims!(0.8, 0.9)
    # savefig("img/MagicTelescopeK.png")
end

# println("Accuracy: ", sum(results), "/", length(results), " = ", mean(results))

# Notes:
#
# EXPERIMENT 1
# Accuracy on non-normalized and normalized.
# Euclidean distance, no distance weighting, k=3
# Non-Normalized : 5388 / 6666 = 0.80828082...
# Normalized (including test set) : 5536 / 6666 = 0.83048...
# Normalized (just by train set) : 5537 / 6666 = 0.830633...
#
# EXPERIMENT 2
# Normalized data, with varying K, no distance weighting, euclidean distance
# img/MagicTelescopeK.png
