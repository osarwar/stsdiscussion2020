using Distributions, StatsBase

function dataGenerator(n, p, sparsity_pattern,
    k = 10, MIC = true, ρ = 0.2, SNR = 10)
    #Define true sparsity
    if sparsity_pattern == "bertsimas"
        nonzeros = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
        if MIC
            β = sample(-1:2:1, k);
        else
            β = [1/sqrt(k) for i in 1:k];
        end
    end
    #Define covariance of predictors
    Σ = eye(p)
    for i in 1:p
        for j in 1:p
            if MIC
                Σ[i,j] = ρ^abs(i-j)
            else
                if i == k + 1 && j <= k
                    Σ[i,j] = 1/(2*k) + 1/(2*sqrt(k))
                elseif j == k + 1 && i <= k
                    Σ[i,j] = 1/(2*k) + 1/(2*sqrt(k))
                end
            end
        end
    end
    #Draw design matrix from Multivariate Normal
    X = rand(MvNormal(zeros(p), Σ), n)'
    #Determine variance of error for specified SNR
    σ² = (β'*Σ[nonzeros,nonzeros]*β)/SNR
    #Draw response
    Y = rand(MvNormal(X[:,nonzeros]*β, σ²*eye(n)))
    #Center the data
    X_train = X[1:Int(n/3),:]; Y_train = Y[1:Int(n/3)];
    X_val = X[Int(n/3)+1:Int(2n/3),:]; Y_val = Y[Int(n/3)+1:Int(2n/3)]
    X_test = X[Int(2n/3)+1:Int(3n/3),:]; Y_test = Y[Int(2n/3)+1:Int(3n/3)]
    for i in 1:p
        X_train[:,i] -= mean(X_train[:,i])
        X_val[:,i] -= mean(X_val[:,i])
        X_test[:,i] -= mean(X_test[:,i])
    end
    Y_train -= mean(Y_train)
    Y_val -= mean(Y_val)
    Y_test -= mean(Y_test)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, nonzeros
end

function rmse(X, Y, β)
    norm(Y + -1*X*β[2:end] + -1*β[1])/sqrt(length(Y))
end

function accuracy(true_support, estimated_support)
    length(intersect(Set(true_support), Set(estimated_support)))/
    length(true_support)
end

function falsePosRate(true_support, estimated_support)
    length(setdiff(Set(estimated_support), Set(true_support)))/
    length(estimated_support)
end

nanmean(x) = mean(filter(!isnan,x))
