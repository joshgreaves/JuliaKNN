module KNN

include("DataPrep.jl")
using .DataPrep.normalizedata

export euclidean_distance, manhattan_distance
export Knn, classify

#==== Distance Functions ====#
function euclidean_distance(point::Matrix{<:Number}, data::Matrix{<:Number})
    sqrt.(sum((point .- data).^2, 2))
end

function manhattan_distance(point::Matrix{<:Number}, data::Matrix{<:Number})
    sum(abs.(point .- data), 2)
end

#==== KNN ====#
struct Knn
    x::Matrix{<:Any}
    y::Matrix{<:Any}
    normalized::Bool
    min_x::Vector{<:Number}
    max_x::Vector{<:Number}
end
function Knn(x::Matrix{<:Any}, y::Matrix{<:Any}, normalize::Bool=true)
    new_x = deepcopy(x)
    new_y = deepcopy(y)
    min_x = Vector{Float64}(size(x)[2])
    max_x = Vector{Float64}(size(x)[2])
    if normalize
        for i in 1:size(x)[2]
            min_x[i] = minimum(x[:, i])
            max_x[i] = maximum(x[:, i])
            new_x[:, i] = normalizedata(x[:, i], min_x[i], max_x[i])
        end
    end
    return Knn(new_x, new_y, normalize, min_x, max_x)
end

function classify(model::Knn, x::Matrix{<:Any}, k::Integer;
                  distance_fn::Function=euclidean_distance,
                  weight_fn::Function=one)
    # The results will be added here
    num_data = size(x)[1]
    num_features = size(x)[2]
    results = Matrix{<:Any}(num_data, 1)

    # If the knn model is normalized, normalize the data
    if model.normalized
        x = deepcopy(x)
        for i in 1:num_features
            x[:, i] = normalizedata(x[:, i], model.min_x[i], model.max_x[i])
        end
    end

    for i in 1:num_data
        # Calculate the distances and get the k closes indices
        distance = distance_fn(x[i:i, :], model.x)
        indices = sortperm(reshape(distance, :))[1:k]

        # Get the classes and distances of the k closes
        actual_classes = model.y[indices]
        distance = distance[indices]

        # Initialize the "votes" for each class
        classes = unique(actual_classes)
        scores = Dict(class => 0.0 for class in classes)
        current_best_class = :none
        current_best_score = 0

        # Loop through each data point and update the votes
        for j in 1:k
            scores[actual_classes[j]] += 1 / weight_fn(distance[j])
            if scores[actual_classes[j]] > current_best_score
                current_best_score = scores[actual_classes[j]]
                current_best_class = actual_classes[j]
            end
        end

        results[i, 1] = current_best_class
    end

    return results
end

function regression(model::Knn, x::Matrix{<:Any}, k::Integer;
                    distance_fn::Function=euclidean_distance,
                    weight_fn::Function=one)

end

function predict(model::Knn, x::Matrix{<:Any})
    # TODO
end

end
