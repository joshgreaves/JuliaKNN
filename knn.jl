module KNN

export Knn, classify

#==== Distance Functions ====#
function euclidean_distance(point::Matrix{<:Any}, data::Matrix{<:Any})
    sqrt.(sum((point .- data).^2, 2))
end

#==== KNN ====#
struct Knn
    x::Matrix{<:Any}
    y::Matrix{<:Any}
end

function classify(model::Knn, x::Matrix{<:Any}, k::Integer;
                  distance_fn::Function=euclidean_distance,
                  weight_fn::Function=one)
    distance = distance_fn(x, model.x)
    indices = sortperm(reshape(distance, :))[1:k]
    results = model.y[indices]
    distance = distance[indices]
    classes = unique(results)
    scores = Dict(class => 0.0 for class in classes)

    current_best_class = :none
    current_best_score = 0
    for i in 1:k
        scores[results[i]] += 1 / weight_fn(distance[i])
        if scores[results[i]] > current_best_score
            current_best_score = scores[results[i]]
            current_best_class = results[i]
        end
    end

    return current_best_class
end

end
