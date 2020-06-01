#lasso======================================================================
# ==========================================================================
function lasso(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    lasso_object = R"lasso($X_train, $Y_train)"
    lasso_regressors = rcopy(R"as.matrix(apply(coef($lasso_object) != 0, 2, which))") #1st regressor is intercept
    #cross-validate to choose best model
    lasso_val_error = rcopy(R"colMeans((predict($lasso_object, newx=$X_val) - $Y_val)^2)")
    best_lambda = indmin(lasso_val_error)
    #final model
    lasso_support= lasso_regressors[best_lambda] + -1  #just array of chosen vars
    if typeof(lasso_support) == Int64
        lasso_support = [lasso_support]
    end
    if 0 in lasso_support #remove intercept
        deleteat!(lasso_support,1)
    end
    β_lasso = rcopy(R"as.matrix(coef($lasso_object))")[:, best_lambda];
    #evaluation metrics
    lasso_RMSE = norm(rcopy(R"as.matrix(predict($lasso_object, $X_test))")[:,best_lambda] + -1*Y_test)/sqrt(size(X_train)[1])
    lasso_Accuracy = accuracy(true_support, lasso_support)
    lasso_FPR  = falsePosRate(true_support, lasso_support)
    # println("True: ", true_support)
    # println("Lasso: ", lasso_support)
    # # println(β_lasso)
    # println(lasso_RMSE, " ", lasso_Accuracy, " ", lasso_FPR, " ")
    return lasso_RMSE, lasso_Accuracy, lasso_FPR
end
# Relaxed lasso=============================================================
# ==========================================================================
function rlasso(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    rlasso_object = R"lasso($X_train, $Y_train, nrelax = 10, nlam=50)"
    rlasso_regressors = rcopy(R"as.matrix(apply(coef($rlasso_object) != 0, 2, which))") #1st regressor is intercept
    #cross-validate to choose best model
    rlasso_val_error = rcopy(R"colMeans((predict($rlasso_object, newx=$X_val) - $Y_val)^2)")
    best_lambda = indmin(rlasso_val_error)
    #final model
    rlasso_support= rlasso_regressors[best_lambda] + -1  #just array of chosen vars
    if typeof(rlasso_support) == Int64
        rlasso_support = [rlasso_support]
    end
    if 0 in rlasso_support #remove intercept
        deleteat!(rlasso_support,1)
    end
    β_rlasso = rcopy(R"as.matrix(coef($rlasso_object))")[:, best_lambda];
    #evaluation metrics
    rlasso_RMSE = norm(rcopy(R"as.matrix(predict($rlasso_object, $X_test))")[:,best_lambda] + -1*Y_test)/sqrt(size(X_test)[1])
    rlasso_Accuracy = accuracy(true_support, rlasso_support)
    rlasso_FPR  = falsePosRate(true_support, rlasso_support)
    # println("Relaxed Lasso: ", rlasso_support)
    # println(β_rlasso)
    # println(rlasso_RMSE, " ", rlasso_Accuracy, " ", rlasso_FPR, " ")
    return rlasso_RMSE, rlasso_Accuracy, rlasso_FPR
end
#MCP========================================================================
# ==========================================================================
function MCP(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    #find cross-validated MCP model using LOOCV
    # n = size(X_train)[1]
    # MCP_cv_object = R"cv.ncvreg($X_train, $Y_train, penalty='MCP', nfolds=$n)"
    # β_MCP = rcopy(R"coef($MCP_cv_object)") #first element is intercept
    MCP_object = R"ncvreg($X_train, $Y_train, penalty='MCP')"
    MCP_val_error = rcopy(R"colMeans((predict($MCP_object, $X_val) - $Y_val)^2)")
    best_lambda = indmin(MCP_val_error)
    #final model
    β_MCP = rcopy(R"as.matrix(coef($MCP_object))")[:, best_lambda]
    MCP_support = find(β_MCP[2:end])
    #evaluation metrics
    # MCP_RMSE = norm(rcopy(R"predict($MCP_cv_object, $X_test)") + -1*Y_test)/sqrt(size(X_train)[1])
    MCP_RMSE = norm(rcopy(R"as.matrix(predict($MCP_object, $X_test))")[:,best_lambda] + -1*Y_test)/sqrt(size(X_test)[1])
    MCP_Accuracy = accuracy(true_support, MCP_support)
    MCP_FPR = falsePosRate(true_support, MCP_support)
    # println("MCP: ", MCP_support)
    # # println(β_MCP)
    # println(MCP_RMSE, " ", MCP_Accuracy, " ", MCP_FPR)
    return MCP_RMSE, MCP_Accuracy, MCP_FPR
end
#SCAD=======================================================================
# ==========================================================================
function SCAD(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    #find cross-validated MCP model using LOOCV
    # n = size(X_train)[1]
    # SCAD_cv_object = R"cv.ncvreg($X_train, $Y_train, penalty='SCAD', nfolds=$n)"
    # β_SCAD = rcopy(R"coef($SCAD_cv_object)") #first element is intercept
    SCAD_object = R"ncvreg($X_train, $Y_train, penalty='SCAD')"
    SCAD_val_error = rcopy(R"colMeans((predict($SCAD_object, $X_val) - $Y_val)^2)")
    best_lambda = indmin(SCAD_val_error)
    #final model
    β_SCAD = rcopy(R"as.matrix(coef($SCAD_object))")[:, best_lambda]
    SCAD_support = find(β_SCAD[2:end])
    #evaluation metrics
    # SCAD_RMSE = norm(rcopy(R"predict($SCAD_cv_object, $X_test)") + -1*Y_test)/sqrt(size(X_train)[1])
    SCAD_RMSE = norm(rcopy(R"as.matrix(predict($SCAD_object, $X_test))")[:,best_lambda] + -1*Y_test)/sqrt(size(X_test)[1])
    SCAD_Accuracy = accuracy(true_support, SCAD_support)
    SCAD_FPR = falsePosRate(true_support, SCAD_support)
    # println("SCAD: ", SCAD_support)
    # println(β_SCAD)
    # println(SCAD_RMSE, " ", SCAD_Accuracy, " ", SCAD_FPR)
    return SCAD_RMSE, SCAD_Accuracy, SCAD_FPR
end
#Forward Stepwise Selection=================================================
# ==========================================================================
function fs(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, maxSteps)
    fs_object = R"fs($X_train, $Y_train, maxsteps = $maxSteps)"
    fs_regressors = rcopy(R"as.matrix(apply(coef($fs_object) != 0, 2, which))")
    #cross-validate to choose best model
    fs_val_error = rcopy(R"colMeans((predict($fs_object, newx=$X_val) - $Y_val)^2)")
    # println(fs_val_error)
    best_step = indmin(fs_val_error)
    # best_step = size(true_support, 1)
    #final model
    fs_support = fs_regressors[best_step] + -1#just array of chosen vars
    if typeof(fs_support) == Int64
        fs_support = [fs_support]
    end
    if 0 in fs_support
        deleteat!(fs_support, 1)
    end
    β_fs = rcopy(R"as.matrix(coef($fs_object))")[:, best_step];
    #evaluation metrics
    fs_RMSE = rmse(X_test, Y_test, β_fs)
    fs_Accuracy = accuracy(true_support, fs_support)
    fs_FPR  = falsePosRate(true_support, fs_support)
    # println("FSS: ", fs_support)
    # println(β_fs)
    # println("fs: ", fs_RMSE, " ", fs_Accuracy, " ", fs_FPR)
    return fs_RMSE, fs_Accuracy, fs_FPR, size(fs_support)[1], fs_object
end
# #Subset Selection CIO =======================================================
# # ===========================================================================
function CIO(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, cv_support_size_delta, min_model_CV, max_model_CV)
    γ0 = 1.*size(X_train,2) / size(true_support,1) / size(X_train,1) / maximum(sum(X_train.^2,2))/2
    CIO_cv_gamma_sizes = [γ0*2*i for i in 8:8+cv_gamma_size_mult]
    CIO_cv_support_sizes = [i for i in min_model_CV:cv_support_size_delta:min(size(X_train)[1],size(X_train)[2], max_model_CV)];
    CIO_support_array = fill([], (size(CIO_cv_gamma_sizes,1), size(CIO_cv_support_sizes,1)));
    CIO_coefficient_array = fill([], (size(CIO_cv_gamma_sizes,1), size(CIO_cv_support_sizes,1)));
    CIO_val_error_array = zeros(size(CIO_cv_gamma_sizes,1), size(CIO_cv_support_sizes,1));
    for γ in enumerate(CIO_cv_gamma_sizes)
        for s in enumerate(CIO_cv_support_sizes)
            CIO_support, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(),
            Y_train, X_train, s[2], γ[2])
            CIO_support_array[γ[1], s[1]] = CIO_support
            CIO_coefficient_array[γ[1], s[1]] = w0
            CIO_val_error_array[γ[1], s[1]] = norm(Y_val + -1*X_val[:,CIO_support]*w0)
        end
    end
    CIO_min_val_idx = ind2sub(size(CIO_val_error_array), indmin(CIO_val_error_array))
    CIO_support = CIO_support_array[CIO_min_val_idx[1], CIO_min_val_idx[2]]
    β_CIO = zeros(size(X_train)[2]+1);β_CIO[1]=0
    for i in enumerate(CIO_support)
        β_CIO[i[2]+1] = CIO_coefficient_array[CIO_min_val_idx[1], CIO_min_val_idx[2]][i[1]]
    end
    CIO_RMSE = rmse(X_test, Y_test, β_CIO)
    CIO_Accuracy = accuracy(true_support, CIO_support)
    CIO_FPR  = falsePosRate(true_support, CIO_support)
    println(CIO_min_val_idx[1])
    # println("CIO: ", CIO_support)
    # println(CIO_RMSE, " ", CIO_Accuracy, " ", CIO_FPR)
    return CIO_RMSE, CIO_Accuracy, CIO_FPR
end
#Subset Selection=======================================================
# ===========================================================================
function SS(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, cv_support_size_delta, min_model_CV, max_model_CV)
    γ0 = 1.*size(X_train,2) / size(true_support,1) / size(X_train,1) / maximum(sum(X_train.^2,2))/2
    SS_cv_gamma_sizes = [γ0*2*i*3 for i in 1:cv_gamma_size_mult]
    SS_cv_support_sizes = [i for i in min_model_CV:cv_support_size_delta:min(size(X_train)[1],size(X_train)[2], max_model_CV)];
    SS_support_array = fill([], (size(SS_cv_gamma_sizes,1), size(SS_cv_support_sizes,1)));
    SS_coefficient_array = fill([], (size(SS_cv_gamma_sizes,1), size(SS_cv_support_sizes,1)));
    SS_val_error_array = zeros(size(SS_cv_gamma_sizes,1), size(SS_cv_support_sizes,1));
    for γ in enumerate(SS_cv_gamma_sizes)
        for s in enumerate(SS_cv_support_sizes)
            SS_object = subsetSelection(OLS(), Constraint(s[2]), Y_train, X_train, δ=1e-4, maxIter=5000)
            SS_support_array[γ[1], s[1]] = SS_object.indices
            SS_coefficient_array[γ[1], s[1]] = SS_object.w
            SS_val_error_array[γ[1], s[1]] = norm(Y_val + -1*X_val[:,SS_object.indices]*SS_object.w)
        end
    end
    SS_min_val_idx = ind2sub(size(SS_val_error_array), indmin(SS_val_error_array))
    SS_support = SS_support_array[SS_min_val_idx[1], SS_min_val_idx[2]]
    β_SS = zeros(size(X_train)[2]+1);β_SS[1]=0
    for i in enumerate(SS_support)
        β_SS[i[2]+1] = SS_coefficient_array[SS_min_val_idx[1], SS_min_val_idx[2]][i[1]]
    end
    SS_RMSE = rmse(X_test, Y_test, β_SS)
    SS_Accuracy = accuracy(true_support, SS_support)
    SS_FPR  = falsePosRate(true_support, SS_support)
    println(SS_min_val_idx[1])
    # println("SS: ", SS_support)
    # println(β_SS)
    println("SS: ", SS_RMSE, " ", SS_Accuracy, " ", SS_FPR)
    return SS_RMSE, SS_Accuracy, SS_FPR
end
#L0Learn=====================================================================
# ===========================================================================
function L0Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    L0Learn_object = R"L0Learn.fit($X_train, $Y_train, algorithm='CDPSI', penalty='L0', nLambda=50)"
    L0Learn_val_error = R"colMeans((as.matrix(predict($L0Learn_object, newx=$X_val))-$Y_val)^2)"
    best_lambda = indmin(L0Learn_val_error)
    β_L0Learn = rcopy(R"as.matrix(coef($L0Learn_object))")[:, best_lambda]
    L0Learn_support = find(β_L0Learn) - 1
    if typeof(L0Learn_support) == Int64
        L0Learn_support = [L0Learn_support]
    end
    if 0 in L0Learn_support #remove intercept
        deleteat!(L0Learn_support,1)
    end
    L0Learn_RMSE = rmse(X_test, Y_test, β_L0Learn)
    L0Learn_Accuracy = accuracy(true_support, L0Learn_support)
    L0Learn_FPR  = falsePosRate(true_support, L0Learn_support)
    # println("L0Learn: ", L0Learn_support)
    # println(β_L0Learn)
    # println("L0Learn: ", L0Learn_RMSE, " ", L0Learn_Accuracy, " ", L0Learn_FPR)
    return L0Learn_RMSE, L0Learn_Accuracy, L0Learn_FPR
end
#L0L1Learn=====================================================================
# ===========================================================================
function L0L1Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    L0L1Learn_object = R"L0Learn.fit($X_train, $Y_train, algorithm='CDPSI', penalty='L0L1', nLambda=50)"
    L0L1Learn_val_error = R"colMeans((as.matrix(predict($L0L1Learn_object, newx=$X_val))-$Y_val)^2)"
    best_lambda = indmin(L0L1Learn_val_error)
    β_L0L1Learn = rcopy(R"as.matrix(coef($L0L1Learn_object))")[:, best_lambda]
    L0L1Learn_support = find(β_L0L1Learn) - 1
    if typeof(L0L1Learn_support) == Int64
        L0L1Learn_support = [L0L1Learn_support]
    end
    if 0 in L0L1Learn_support #remove intercept
        deleteat!(L0L1Learn_support,1)
    end
    L0L1Learn_RMSE = rmse(X_test, Y_test, β_L0L1Learn)
    L0L1Learn_Accuracy = accuracy(true_support, L0L1Learn_support)
    L0L1Learn_FPR  = falsePosRate(true_support, L0L1Learn_support)
    # println("L0L1Learn: ", L0L1Learn_support)
    # println(β_L0L1Learn)
    # println("L0L1Learn: ", L0L1Learn_RMSE, " ", L0L1Learn_Accuracy, " ", L0L1Learn_FPR)
    return L0L1Learn_RMSE, L0L1Learn_Accuracy, L0L1Learn_FPR
end
#L0L2Learn=====================================================================
# ===========================================================================
function L0L2Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support)
    L0L2Learn_object = R"L0Learn.fit($X_train, $Y_train, algorithm='CDPSI', penalty='L0L2', nLambda=50)"
    L0L2Learn_val_error = R"colMeans((as.matrix(predict($L0L2Learn_object, newx=$X_val))-$Y_val)^2)"
    best_lambda = indmin(L0L2Learn_val_error)
    β_L0L2Learn = rcopy(R"as.matrix(coef($L0L2Learn_object))")[:, best_lambda]
    L0L2Learn_support = find(β_L0L2Learn) - 1
    if typeof(L0L2Learn_support) == Int64
        L0L2Learn_support = [L0L2Learn_support]
    end
    if 0 in L0L2Learn_support #remove intercept
        deleteat!(L0L2Learn_support,1)
    end
    L0L2Learn_RMSE = rmse(X_test, Y_test, β_L0L2Learn)
    L0L2Learn_Accuracy = accuracy(true_support, L0L2Learn_support)
    L0L2Learn_FPR  = falsePosRate(true_support, L0L2Learn_support)
    # println("L0L2Learn: ", L0L2Learn_support)
    # println(β_L0L2Learn)
    # println("L0L2Learn: ", L0L2Learn_RMSE, " ", L0L2Learn_Accuracy, " ", L0L2Learn_FPR)
    return L0L2Learn_RMSE, L0L2Learn_Accuracy, L0L2Learn_FPR
end
