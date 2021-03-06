module DataPrep

export partition, splitdata, shuffledata, getbatch, remove_unknowns, normalizedata!

function remove_unknowns(x::Matrix{<:Any}, y::Matrix{<:Any}; unk::Any=:?)
    x = copy(x)

    classes = unique(y)
    num_data = size(x)[1]
    num_features = size(x)[2]

    bests = Dict{Symbol, Vector{Any}}()
    # Collect the mean for each class
    for class in classes
        indices = y[:, 1] .== class
        class_xs = x[indices, :]

        best = Vector{Any}(num_features)
        for i in 1:num_features
            if typeof(class_xs[1, i]) == Symbol
                counts = Dict{Symbol, Int64}()
                features = unique(class_xs[:, i])
                for feature in features
                    counts[feature] = sum(class_xs[:, i] .== feature)
                end

                best_s = :none
                best_n = 0
                for key in keys(counts)
                    if counts[key] > best_n && key != unk
                        best_s = key
                        best_n = counts[key]
                    end
                end

                best[i] = best_s
            else
                val_indices = issubtype.(typeof.(class_xs[:, i]), Number)
                mean_val = mean(class_xs[val_indices, i])
                best[i] = mean_val
            end
        end
        bests[class] = best
    end

    # Now we have the bests, loop through the data altering it
    for i in 1:num_data
        for j in 1:num_features
            if x[i, j] == unk
                x[i, j] = bests[y[i, 1]][j]
            end
        end
    end

    return x
end

function remove_unknowns(x::Vector{<:Any}, vals::Vector{<:Any})
    mean_val = mean(vals)
    x[typeof.(x) .== Symbol] = mean_val
    return x
end

function partition(x::Matrix{<:Any}, y::Matrix{<:Any}, n::Integer)
    num_data = size(x)[1]
    partition_size = num_data / n
    xs = Vector{typeof(x)}(n)
    ys = Vector{typeof(y)}(n)
    last = 1
    for i in 1:n
        next = Int32(floor(partition_size * i))
        xs[i] = x[last:next, :]
        ys[i] = y[last:next, :]
        last = next + 1
    end
    return xs, ys
end

function splitdata(data::Array{<:Any, 2}, split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return (data[indices[1:split_index],:], data[indices[(split_index+1):end],:])
end

function splitdata(data::Array{<:Any, 2}, labels::Array{<:Any, 2},
                   split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return data[indices[1:split_index],:], labels[indices[1:split_index],:],
        data[indices[(split_index+1):end],:], labels[indices[(split_index+1):end],:]
end

function shuffledata(data::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 1})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices,:]
end

function getbatch(data::Matrix{<:Any}, labels::Matrix{<:Any};
                  batch_size::Integer=32)
    num_data = size(data)[1]
    indices = shuffle(1:num_data)
    return data[indices[1:batch_size], :], labels[indices[1:batch_size], :]
end

function smoothline(data::Array{<:Number, 1}; window=10)
    result = Array{Float64, 1}(Int32(floor(size(data)[1] / 10)))
    for i in 1:length(result)
        result[i] = mean(data[((i-1)*window+1):(i*window)])
    end
    return result
end

function normalizedata(data::Vector{<:Number}, actual_lower::Number,
                       actual_upper::Number, target_lower::Number=0,
                       target_upper::Number=1)
    actual_span = actual_upper - actual_lower
    target_span = target_upper - target_lower

    data .-= actual_lower
    data .*= (target_span / actual_span)
    data .+= target_lower
    return data
end

function normalizedata!(data1::Matrix{<:Number}, data2::Matrix{<:Number},
                   lower::Number=0, upper::Number=1)
    # Needs to be normalized by column
    for i in 1:size(data1)[2]
        lb = min(minimum(data1[:, i]), minimum(data2[:, i]))
        ub = max(maximum(data1[:, i]), maximum(data2[:, i]))
        data1[:, i] = normalizedata(data1[:, i], lb, ub, lower, upper)
        data2[:, i] = normalizedata(data2[:, i], lb, ub, lower, upper)
    end
end

end
