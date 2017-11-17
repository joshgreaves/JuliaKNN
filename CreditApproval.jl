include("Arff.jl")
include("knn.jl")
include("DataPrep.jl")

using Plots

# push!(LOAD_PATH, ".")

importall Arff
importall KNN
importall DataPrep

arff = loadarff("data/CreditApproval.arff")
x = arff.data[:, 1:end-1]
y = arff.data[:, end:end]
train_x, train_y, test_x, test_y = splitdata(x, y)
train_x = remove_unknowns(train_x, train_y)

nominal_indices = [1, 4, 5, 6, 7, 9, 10, 12, 13]
for i in 1:size(test_x)[2]
    if !(i in nominal_indices)
        test_x[:, i] = remove_unknowns(test_x[:, i], train_x[:, i])
    end
end

model = Knn(train_x, train_y, vdm=true,
            nominal_indices=nominal_indices)

# Graph for all odd values between k = 1 and k = 15
accuracy = Vector{Float64}(8)
for i = 1:8
    println(i)
    predictions = classify(model, test_x, (i*2) - 1, distance_fn=vdm,
                           weight_fn=f(x)=x^2)
    accuracy[i] = mean(predictions .== test_y)
end

function plotacc()
    pyplot()
    plot(accuracy, xticks=(1:8, 1:2:15), label="Test Accuracy")
    title!("Accuracy on Credit Approval Dataset")
    xaxis!("k")
    yaxis!("Accuracy")
    # savefig("img/CreditApproval.png")
end

# Notes:
