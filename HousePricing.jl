include("Arff.jl")
include("knn.jl")

using Plots

# push!(LOAD_PATH, ".")

importall Arff
importall KNN

train_arff = loadarff("data/HousePricingTrain.arff")
test_arff = loadarff("data/HousePricingTest.arff")

train_x = convert(Matrix{Float64}, train_arff.data[:, 1:end-1])
train_y = convert(Matrix{Float64}, train_arff.data[:, end:end])
test_x = convert(Matrix{Float64}, test_arff.data[:, 1:end-1])
test_y = convert(Matrix{Float64}, test_arff.data[:, end:end])

model = Knn(train_x, train_y)

# Graph for all odd values between k = 1 and k = 15
mse = Vector{Float64}(8)
for i = 1:8
    println(i)
    predictions = regression(model, test_x, (i*2) - 1)
    # predictions = regression(model, test_x, (i*2) - 1, weight_fn=f(x)=x^2)
    diff = predictions - test_y
    diff_2 = diff.^2
    mse[i] = mean(diff_2)
end

function plotmse()
    pyplot()
    plot(mse, xticks=(1:8, 1:2:15), label="Test MSE")
    title!("MSE on House Pricing Dataset with Varying k")
    xaxis!("k")
    yaxis!("MSE")
    # ylims!(15, 25)
    # savefig("img/HousePricingMSEWithWeighting.png")
end

# Notes:
# EXPERIMENT 1
# KNN on this dataset without distance weighting with variable k
# Saved as "img/HousePricingMSE.png"
#
# EXPERIMENT 2
# KNN on the dataset with squared distance weighting with variable k
# Saved as "img/HousePricingMSEWithWeighting.png"
