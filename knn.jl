module KNN

include("DataPrep.jl")
using .DataPrep.normalizedata

export euclidean_distance, manhattan_distance, vdm
export Knn, classify, regression

#==== VDM ====#
struct VDMData
    on::Bool
    indices::Vector{<:Integer}
    n_ax::Dict
    n_axc::Dict
    c::Dict
end
function VDMData()
    VDMData(false, Vecotr{Int32}(), Dict(), Dict(), Dict())
end

#==== KNN ====#
struct Knn
    x::Matrix{<:Any}
    y::Matrix{<:Any}
    normalized::Bool
    min_x::Vector{<:Number}
    max_x::Vector{<:Number}
    vdmdata::VDMData
end
function Knn(x::Matrix{<:Any}, y::Matrix{<:Any}, normalize::Bool=true;
             vdm::Bool=false,
             nominal_indices::Vector{<:Integer}=Vector{Int64}())
    new_x = deepcopy(x)
    new_y = deepcopy(y)
    min_x = Vector{Float64}(size(x)[2])
    max_x = Vector{Float64}(size(x)[2])

    vdm_indices = nominal_indices
    vdm_n_ax = Dict()
    vdm_n_axc = Dict()
    vdm_c = Dict()

    classes = unique(y)
    for class in classes
        vdm_c[class] = 1
    end

    if normalize || vdm
        for i in 1:size(x)[2]
            if normalize && !(i in nominal_indices)
                min_x[i] = minimum(x[:, i])
                max_x[i] = maximum(x[:, i])
                new_x[:, i] = normalizedata(convert(Vector{Float64}, x[:, i]), min_x[i], max_x[i])
            elseif vdm && i in nominal_indices
                push!(vdm_indices, i)
                feature_names = unique(x[:, i])
                for feature in feature_names
                    ax = (i, feature)
                    vdm_n_ax[ax] = 1 # Laplassian smoothing
                    for class in classes
                        axc = (i, feature, class)
                        vdm_n_axc[axc] = 1 # Laplassian smoothing
                    end
                end
                for j in 1:size(x)[1]
                    feature = x[j, i]
                    class = y[j, 1]
                    vdm_n_ax[(i, feature)] += 1
                    vdm_n_axc[(i, feature, class)] += 1
                    vdm_c[class] += 1
                end

                # For mising values
                vdm_n_ax[(i, :?)] = 1
                for class in classes
                    vdm_n_axc[(i, :?, class)] = 1
                end
            end
        end
    end
    return Knn(new_x, new_y, normalize, min_x, max_x,
               VDMData(vdm, vdm_indices, vdm_n_ax, vdm_n_axc, vdm_c))
end

#==== Distance Functions ====#
function euclidean_distance(point::Matrix{<:Number}, knn::Knn)
    data = knn.x
    sqrt.(sum((point .- data).^2, 2))
end

function vdm(sym1::Symbol, sym2::Symbol, index::Integer, vdmdata::VDMData)
    total = 0
    for class in keys(vdmdata.c)
        lhs = vdmdata.n_axc[index, sym1, class] / vdmdata.n_ax[index, sym1]
        rhs = vdmdata.n_axc[index, sym2, class] / vdmdata.n_ax[index, sym2]
        total += (lhs - rhs)^2
    end
    return total
end

function vdm(point::Matrix{<:Any}, knn::Knn)
    data = knn.x
    num_data, num_features = size(data)
    result = Matrix{Float64}(num_data, num_features)
    for j in 1:num_features
        if j in knn.vdmdata.indices
            for i in 1:num_data
                result[i, j] = vdm(point[1, j], data[i, j], j, knn.vdmdata)
            end
        else
            result[:, j] = (point[1, j] .- data[:, j]).^2 # squared diff
        end
    end
    return sqrt.(sum(result, 2))
end

function manhattan_distance(point::Matrix{<:Number}, knn::Knn)
    data = knn.x
    sum(abs.(point .- data), 2)
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
            if !(i in model.vdmdata.indices)
                x[:, i] = normalizedata(convert(Vector{Float64}, x[:, i]),
                                        model.min_x[i], model.max_x[i])
            end
        end
    end

    for i in 1:num_data
        # Calculate the distances and get the k closes indices
        distance = distance_fn(x[i:i, :], model)
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
          distance = distance_fn(x[i:i, :], model)
          indices = sortperm(reshape(distance, :))[1:k]

          # Get the classes and distances of the k closes
          y = model.y[indices]
          distance = distance[indices]

          # Initialize the "votes" for each class
          total_val = 0
          total_weight = 0

          # Loop through each data point and update the votes
          for j in 1:k
              distance_weight = weight_fn(distance[j])
              total_val += y[j] / distance_weight
              total_weight += 1 / weight_fn(distance[j])
          end

          results[i, 1] = total_val / total_weight
      end

      return results
end

function predict(model::Knn, x::Matrix{<:Any})
    # TODO
end

end
