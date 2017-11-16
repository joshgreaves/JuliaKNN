include("Arff.jl")
include("knn.jl")
include("DataPrep.jl")

importall Arff
importall KNN
importall DataPrep

train_arff = loadarff("data/MagicTelescopeTrain.arff")
test_arff = loadarff("data/MagicTelescopeTest.arff")

train_x = train_arff.data[:, 1:end-1]
train_y = train_arff.data[:, end:end]
test_x = test_arff.data[:, 1:end-1]
test_y = test_arff.data[:, end:end]

println(maximum(train_x))
normalize!(train_x, test_x)
println(maximum(train_x))

model = Knn(train_x, train_y)

# Classify
count = 0
total = size(test_y)[1]
for i in 1:total
    println(i, ": ", total)
    count += classify(model, test_x[i:i, :], 3) == test_y[i, 1]
end

println("Accuracy: ", count, "/", total, " = ", count/total)

# Non-Normalized : 5388 / 6666 = 0.80828082...
# Normalized : 
