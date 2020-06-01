t = true
f = false
test_lasso = t; test_rlasso = t;
test_MCP = t; test_SCAD = f;
test_fs = t;
test_CIO = f; test_SS = f;
test_L0Learn = t; test_L0L1Learn = t; test_L0L2Learn = t;

if test_lasso || test_rlasso || test_MCP || test_SCAD || test_fs || test_L0Learn; using RCall; end
if test_fs; maxSteps = 100; end
if test_CIO; using SubsetSelectionCIO; end
if test_SS; using SubsetSelection; end
if test_SS || test_CIO; cv_gamma_size_mult = 10; cv_support_size_delta = 1; min_model_CV = 50; max_model_CV = 50; end
if (test_CIO || test_SS) && test_fs; cv_based_on_fs = f; window = 0; else; cv_based_on_fs = f; end
if test_lasso || test_rlasso || test_fs; R"library(bestsubset)"; end
if test_MCP || test_SCAD; R"library(ncvreg)"; end
if test_L0Learn || test_L0L1Learn || test_L0L2Learn; R"library(L0Learn)"; end

using TimerOutputs
const to = TimerOutput()
using JLD
include("test_utils.jl")
include("regression_methods.jl")

#problem specification
p=Int(1e4); k=50; sparsity_pattern = "bertsimas"; MIC=true; ρ=0.7; SNR=1;

path = string(pwd(), "\\VarN_", "k_", k, "p_", p, "ρ_", ρ,
"MIC_", MIC, "SNR_", SNR, "pattern_", sparsity_pattern)
mkdir(path)

num_n = 16; n_min = 500; n_max = 2000; probs_per_n = 10;

#Arrays for storing metric per size of n
if test_lasso; lasso_Accuracy_n = Float64[]; lasso_FPR_n  = Float64[]; lasso_RMSE_n  = Float64[]; end
if test_rlasso; rlasso_Accuracy_n  = Float64[]; rlasso_FPR_n  = Float64[]; rlasso_RMSE_n  = Float64[]; end
if test_MCP; MCP_Accuracy_n  = Float64[]; MCP_FPR_n  = Float64[]; MCP_RMSE_n  = Float64[]; end
if test_SCAD; SCAD_Accuracy_n  = Float64[]; SCAD_FPR_n  = Float64[]; SCAD_RMSE_n  = Float64[]; end
if test_fs; fs_Accuracy_n  = Float64[]; fs_FPR_n  = Float64[]; fs_RMSE_n  = Float64[]; end
if test_CIO; CIO_Accuracy_n  = Float64[]; CIO_FPR_n  = Float64[]; CIO_RMSE_n  = Float64[]; end
if test_SS; SS_Accuracy_n  = Float64[]; SS_FPR_n  = Float64[]; SS_RMSE_n  = Float64[]; end
if test_L0Learn; L0Learn_Accuracy_n  = Float64[]; L0Learn_FPR_n  = Float64[]; L0Learn_RMSE_n  = Float64[]; end
if test_L0L1Learn; L0L1Learn_Accuracy_n  = Float64[]; L0L1Learn_FPR_n  = Float64[]; L0L1Learn_RMSE_n  = Float64[]; end
if test_L0L2Learn; L0L2Learn_Accuracy_n  = Float64[]; L0L2Learn_FPR_n  = Float64[]; L0L2Learn_RMSE_n  = Float64[]; end
N = Int32[]

for n in linspace(n_min, n_max, num_n)
    @timeit to string("N: ", string(Int(floor(n)))) begin
    append!(N, Int(floor(n)))
    println("n: $n\n")
    results = open(string(path, "\\", Int(floor(n)), ".txt"), "a")
    write(results, string("Number of data points:", Int(floor(n)), "\n"))
    #arrays for storing metrics for each iteration of size n
    if test_lasso; lasso_Accuracy_array = Float64[]; lasso_FPR_array = Float64[]; lasso_RMSE_array = Float64[]; end
    if test_rlasso; rlasso_Accuracy_array = Float64[]; rlasso_FPR_array = Float64[]; rlasso_RMSE_array = Float64[]; end
    if test_MCP; MCP_Accuracy_array = Float64[]; MCP_FPR_array = Float64[]; MCP_RMSE_array = Float64[]; end
    if test_SCAD; SCAD_Accuracy_array = Float64[]; SCAD_FPR_array = Float64[]; SCAD_RMSE_array = Float64[]; end
    if test_fs; fs_Accuracy_array = Float64[]; fs_FPR_array = Float64[]; fs_RMSE_array = Float64[]; end
    if test_CIO; CIO_Accuracy_array = Float64[]; CIO_FPR_array = Float64[]; CIO_RMSE_array = Float64[]; end
    if test_SS; SS_Accuracy_array = Float64[]; SS_FPR_array = Float64[]; SS_RMSE_array = Float64[]; end
    if test_L0Learn; L0Learn_Accuracy_array = Float64[]; L0Learn_FPR_array = Float64[]; L0Learn_RMSE_array = Float64[]; end
    if test_L0L1Learn; L0L1Learn_Accuracy_array = Float64[]; L0L1Learn_FPR_array = Float64[]; L0L1Learn_RMSE_array = Float64[]; end
    if test_L0L2Learn; L0L2Learn_Accuracy_array = Float64[]; L0L2Learn_FPR_array = Float64[]; L0L2Learn_RMSE_array = Float64[]; end

    for prob in 1:probs_per_n
        println("instance: $prob\n")
        #training, validation, test data for each n
        X_train,  X_val,  X_test,  Y_train,  Y_val,  Y_test,  true_support =
        dataGenerator(3Int(floor(n)),p,sparsity_pattern,k, MIC, ρ, SNR)
        # write(results, string("X_train\n", X_train, "\n", "X_val\n", X_val, "\n", "X_test\n", X_test, "\n",
        # "Y_train\n", Y_train, "\n", "Y_val\n", Y_val, "\n", "Y_test\n", Y_test, "\n"))
        # write(results, "TRUE SUPPORT: $true_support \n")
        write(results, string("Instance ", prob, "\n"))
        write(results, "Method Accuracy FPR RMSE \n")
        if test_lasso; println("lasso"); @timeit to "lasso" lasso_RMSE, lasso_Accuracy, lasso_FPR =
            lasso(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(lasso_Accuracy_array, lasso_Accuracy); append!(lasso_FPR_array, lasso_FPR); append!(lasso_RMSE_array, lasso_RMSE);
            write(results, "Lasso       $lasso_Accuracy $lasso_FPR $lasso_RMSE\n"); end
        if test_rlasso; println("rlasso"); @timeit to "rlasso" rlasso_RMSE, rlasso_Accuracy, rlasso_FPR =
            rlasso(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(rlasso_Accuracy_array, rlasso_Accuracy); append!(rlasso_FPR_array, rlasso_FPR); append!(rlasso_RMSE_array, rlasso_RMSE);
            write(results, "rLasso      $rlasso_Accuracy $rlasso_FPR $rlasso_RMSE\n"); end
        if test_MCP; println("MCP"); @timeit to "MCP" MCP_RMSE, MCP_Accuracy, MCP_FPR =
            MCP(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(MCP_Accuracy_array, MCP_Accuracy); append!(MCP_FPR_array, MCP_FPR); append!(MCP_RMSE_array, MCP_RMSE);
            write(results, "MCP         $MCP_Accuracy $MCP_FPR $MCP_RMSE\n"); end
        if test_SCAD; println("SCAD"); @timeit to "SCAD" SCAD_RMSE, SCAD_Accuracy, SCAD_FPR =
            SCAD(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(SCAD_Accuracy_array, SCAD_Accuracy); append!(SCAD_FPR_array, SCAD_FPR); append!(SCAD_RMSE_array, SCAD_RMSE);
            write(results, "SCAD        $SCAD_Accuracy $SCAD_FPR $SCAD_RMSE\n"); end
        if test_fs; println("fs"); @timeit to "fs" fs_RMSE, fs_Accuracy, fs_FPR, fs_support_size, fs_object =
            fs(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, maxSteps);
            append!(fs_Accuracy_array, fs_Accuracy); append!(fs_FPR_array, fs_FPR); append!(fs_RMSE_array, fs_RMSE);
            write(results, "fs          $fs_Accuracy $fs_FPR $fs_RMSE\n");
            if fs_support_size < 10; cv_based_on_fs = false; else; cv_based_on_fs = true; end; end
        if test_CIO; println("CIO");
            if cv_based_on_fs; min_model_CV = fs_support_size - window; max_model_CV = fs_support_size + window;
            @timeit to "CIO" CIO_RMSE, CIO_Accuracy, CIO_FPR =
            CIO(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, 1, min_model_CV, max_model_CV);
        else; @timeit to "CIO" CIO_RMSE, CIO_Accuracy, CIO_FPR =
            CIO(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, cv_support_size_delta, min_model_CV, max_model_CV); end
            append!(CIO_Accuracy_array, CIO_Accuracy); append!(CIO_FPR_array, CIO_FPR); append!(CIO_RMSE_array, CIO_RMSE);
            write(results, "CIO       $CIO_Accuracy $CIO_FPR $CIO_RMSE\n"); end
        if test_SS; println("SS");
            if cv_based_on_fs; min_model_CV = fs_support_size - window; max_model_CV = fs_support_size + window;
            @timeit to "SS" SS_RMSE, SS_Accuracy, SS_FPR =
            SS(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, 1, min_model_CV, max_model_CV);
            else; @time SS_RMSE, SS_Accuracy, SS_FPR =
            SS(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support, cv_gamma_size_mult, cv_support_size_delta, min_model_CV, max_model_CV); end
            append!(SS_Accuracy_array, SS_Accuracy); append!(SS_FPR_array, SS_FPR); append!(SS_RMSE_array, SS_RMSE);
            write(results, "SS          $SS_Accuracy $SS_FPR $SS_RMSE\n"); end;
        if test_L0Learn; println("L0Learn");
            @timeit to "L0Learn" L0Learn_RMSE, L0Learn_Accuracy, L0Learn_FPR =
            L0Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(L0Learn_Accuracy_array, L0Learn_Accuracy); append!(L0Learn_FPR_array, L0Learn_FPR); append!(L0Learn_RMSE_array, L0Learn_RMSE);
            write(results, "L0Learn     $L0Learn_Accuracy $L0Learn_FPR $L0Learn_RMSE\n"); end;
        if test_L0L1Learn; println("L0L1Learn");
            @timeit to "L0L1Learn" L0L1Learn_RMSE, L0L1Learn_Accuracy, L0L1Learn_FPR =
            L0L1Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(L0L1Learn_Accuracy_array, L0L1Learn_Accuracy); append!(L0L1Learn_FPR_array, L0L1Learn_FPR); append!(L0L1Learn_RMSE_array, L0L1Learn_RMSE);
            write(results, "L0L1Learn   $L0L1Learn_Accuracy $L0L1Learn_FPR $L0L1Learn_RMSE\n"); end;
        if test_L0L2Learn; println("L0L2Learn");
            @timeit to "L0L2Learn" L0L2Learn_RMSE, L0L2Learn_Accuracy, L0L2Learn_FPR =
            L0L2Learn(X_train, X_val, X_test, Y_train, Y_val, Y_test, true_support);
            append!(L0L2Learn_Accuracy_array, L0L2Learn_Accuracy); append!(L0L2Learn_FPR_array, L0L2Learn_FPR); append!(L0L2Learn_RMSE_array, L0L2Learn_RMSE);
            write(results, "L0L2Learn   $L0L2Learn_Accuracy $L0L2Learn_FPR $L0L2Learn_RMSE\n"); end;
    end

    write(results,"Average\n Method Accuracy FPR RMSE\n")
    if test_lasso; append!(lasso_Accuracy_n, nanmean(lasso_Accuracy_array));
        append!(lasso_FPR_n , nanmean(lasso_FPR_array)); append!(lasso_RMSE_n , nanmean(lasso_RMSE_array));
        write(results, string("Lasso         ", lasso_Accuracy_n[end], " ", lasso_FPR_n[end], " ", lasso_RMSE_n[end], "\n")); end
    if test_rlasso; append!(rlasso_Accuracy_n , nanmean(rlasso_Accuracy_array));
        append!(rlasso_FPR_n , nanmean(rlasso_FPR_array)); append!(rlasso_RMSE_n , nanmean(rlasso_RMSE_array));
        write(results, string("rLasso        ", rlasso_Accuracy_n[end], " ", rlasso_FPR_n[end], " ", rlasso_RMSE_n[end], "\n")); end
    if test_MCP; append!(MCP_Accuracy_n , nanmean(MCP_Accuracy_array));
        append!(MCP_FPR_n , nanmean(MCP_FPR_array)); append!(MCP_RMSE_n , nanmean(MCP_RMSE_array));
        write(results, string("MCP           ", MCP_Accuracy_n[end], " ", MCP_FPR_n[end], " ", MCP_RMSE_n[end], "\n")); end
    if test_SCAD; append!(SCAD_Accuracy_n , nanmean(SCAD_Accuracy_array));
        append!(SCAD_FPR_n , nanmean(SCAD_FPR_array)); append!(SCAD_RMSE_n , nanmean(SCAD_RMSE_array));
        write(results, string("SCAD          ", SCAD_Accuracy_n[end], " ", SCAD_FPR_n[end], " ", SCAD_RMSE_n[end], "\n")); end
    if test_fs; append!(fs_Accuracy_n , nanmean(fs_Accuracy_array));
        append!(fs_FPR_n , nanmean(fs_FPR_array)); append!(fs_RMSE_n , nanmean(fs_RMSE_array));
        write(results, string("fs            ", fs_Accuracy_n[end], " ", fs_FPR_n[end], " ", fs_RMSE_n[end], "\n")); end
    if test_CIO; append!(CIO_Accuracy_n , nanmean(CIO_Accuracy_array));
        append!(CIO_FPR_n , nanmean(CIO_FPR_array)); append!(CIO_RMSE_n , nanmean(CIO_RMSE_array));
        write(results, string("CIO           ", CIO_Accuracy_n[end], " ", CIO_FPR_n[end], " ", CIO_RMSE_n[end], "\n")); end
    if test_SS; append!(SS_Accuracy_n , nanmean(SS_Accuracy_array));
        append!(SS_FPR_n , nanmean(SS_FPR_array)); append!(SS_RMSE_n , nanmean(SS_RMSE_array));
        write(results, string("SS            ", SS_Accuracy_n[end], " ", SS_FPR_n[end], " ", SS_RMSE_n[end], "\n")); end
    if test_L0Learn; append!(L0Learn_Accuracy_n , nanmean(L0Learn_Accuracy_array));
        append!(L0Learn_FPR_n , nanmean(L0Learn_FPR_array)); append!(L0Learn_RMSE_n , nanmean(L0Learn_RMSE_array));
        write(results, string("L0Learn       ", L0Learn_Accuracy_n[end], " ", L0Learn_FPR_n[end], " ", L0Learn_RMSE_n[end], "\n")); end
    if test_L0L1Learn; append!(L0L1Learn_Accuracy_n , nanmean(L0L1Learn_Accuracy_array));
        append!(L0L1Learn_FPR_n , nanmean(L0L1Learn_FPR_array)); append!(L0L1Learn_RMSE_n , nanmean(L0L1Learn_RMSE_array));
        write(results, string("L0L1Learn     ", L0L1Learn_Accuracy_n[end], " ", L0L1Learn_FPR_n[end], " ", L0L1Learn_RMSE_n[end], "\n")); end
    if test_L0L2Learn; append!(L0L2Learn_Accuracy_n , nanmean(L0L2Learn_Accuracy_array));
        append!(L0L2Learn_FPR_n , nanmean(L0L2Learn_FPR_array)); append!(L0L2Learn_RMSE_n , nanmean(L0L2Learn_RMSE_array));
        write(results, string("L0L2Learn     ", L0L2Learn_Accuracy_n[end], " ", L0L2Learn_FPR_n[end], " ", L0L2Learn_RMSE_n[end], "\n")); end
    end
    close(results)
    @save string("VarN_dataArrays_SNR", string(SNR), ".jld")
end



using Plots

Accuracy_plot = plot(title="Accuracy", xlabel="Number of data points", ylabel ="Accuracy %", legend = false)
FPR_plot = plot(title="FPR", xlabel="Number of data points", ylabel ="FPR %", legend = false)
RMSE_plot = plot(title="RMSE", xlabel="Number of data points", ylabel ="RMSE", legend = false)

if test_lasso; plot!(Accuracy_plot, N, lasso_Accuracy_n*100, label="lasso", lw=2, color=:hotpink);
plot!(FPR_plot, N, lasso_FPR_n*100, label="lasso", lw=2, color=:hotpink);
plot!(RMSE_plot, N, lasso_RMSE_n, label="lasso", lw=2, color=:hotpink); end

if test_rlasso; plot!(Accuracy_plot, N, rlasso_Accuracy_n*100, label="rlasso", lw=2, color=:purple);
plot!(FPR_plot, N, rlasso_FPR_n*100, label="rlasso", lw=2, color=:purple);
plot!(RMSE_plot, N, rlasso_RMSE_n, label="rlasso", lw=2, color=:purple); end

if test_MCP; plot!(Accuracy_plot, N, MCP_Accuracy_n*100, label="MCP", lw=2, color=:orange);
plot!(FPR_plot, N, MCP_FPR_n*100, label="MCP", lw=2, color=:orange);
plot!(RMSE_plot, N, MCP_RMSE_n, label="MCP", lw=2, color=:orange); end

if test_SCAD; plot!(Accuracy_plot, N, SCAD_Accuracy_n*100, label="SCAD", lw=2, color=:pink);
plot!(FPR_plot, N, SCAD_FPR_n*100, label="SCAD", lw=2, color=:pink);
plot!(RMSE_plot, N, SCAD_RMSE_n, label="SCAD", lw=2, color=:pink); end

if test_fs; plot!(Accuracy_plot, N, fs_Accuracy_n*100, label="fs", lw=2, color=:deepskyblue1);
plot!(FPR_plot, N, fs_FPR_n*100, label="fs", lw=2, color=:deepskyblue1);
plot!(RMSE_plot, N, fs_RMSE_n, label="fs", lw=2, color=:deepskyblue1); end

if test_CIO; plot!(Accuracy_plot, N, CIO_Accuracy_n*100, label="CIO", lw=2, color=:green);
plot!(FPR_plot, N, CIO_FPR_n*100, label="CIO", lw=2, color=:green);
plot!(RMSE_plot, N, CIO_RMSE_n, label="CIO", lw=2, color=:green); end

if test_SS; plot!(Accuracy_plot, N, SS_Accuracy_n*100, label="SS", lw=2, color=:darkseagreen);
plot!(FPR_plot, N, SS_FPR_n*100, label="SS", lw=2, color=:darkseagreen);
plot!(RMSE_plot, N, SS_RMSE_n, label="SS", lw=2, color=:darkseagreen); end

if test_L0Learn; plot!(Accuracy_plot, N, L0Learn_Accuracy_n*100, label="L0Learn", lw=2, color=:orchid1);
plot!(FPR_plot, N, L0Learn_FPR_n*100, label="L0Learn", lw=2, color=:orchid1);
plot!(RMSE_plot, N, L0Learn_RMSE_n, label="L0Learn", lw=2, color=:orchid1); end

if test_L0L1Learn; plot!(Accuracy_plot, N, L0L1Learn_Accuracy_n*100, label="L0L1Learn", lw=2, color=:mediumorchid);
plot!(FPR_plot, N, L0L1Learn_FPR_n*100, label="L0L1Learn", lw=2, color=:mediumorchid);
plot!(RMSE_plot, N, L0L1Learn_RMSE_n, label="L0L1Learn", lw=2, color=:mediumorchid); end

if test_L0L2Learn; plot!(Accuracy_plot, N, L0L2Learn_Accuracy_n*100, label="L0L2Learn", lw=2, color=:darkorchid);
plot!(FPR_plot, N, L0L2Learn_FPR_n*100, label="L0L2Learn", lw=2, color=:darkorchid);
plot!(RMSE_plot, N, L0L2Learn_RMSE_n, label="L0L2Learn", lw=2, color=:darkorchid); end

savefig(Accuracy_plot, string(path, "\\Accuracy_plot.png"))
savefig(FPR_plot, string(path, "\\FPR_plot.png"))
savefig(RMSE_plot, string(path, "\\RMSE_plot.png"))
