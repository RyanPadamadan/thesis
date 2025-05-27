import matplotlib.pyplot as plt
from algorithms import (
    find_best_experimental,
    k_means,
    k_means_ewma,
    k_medoid,
    median,
    weighted_mean
)
from spatial_manager import assign_weights, assign_weights_rssi
from data_cleaner import *

def run_all_algorithms(exp_dir, k=4, alpha=0.7, prev=None):
    """
    Runs each algorithm on the given experiment and returns their errors.

    Returns:
        dict: {algorithm_name: error}
    """
    errors = {}

    # k-means
    try:
        final, device = find_best_experimental(exp_dir, k_means, k)
        errors["k-means"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - k-means failed: {e}")
        errors["k-means"] = None

    # k-means EWMA
    # try:
    #     final, device = find_best_experimental(exp_dir, k_means_ewma, k, it=True, prev=prev)
    #     errors["k-means EWMA"] = distance(final, device)
    # except Exception as e:
    #     print(f"{exp_dir} - k-means EWMA failed: {e}")
    #     errors["k-means EWMA"] = None

    # median
    try:
        final, device = find_best_experimental(exp_dir, median, k)
        errors["median"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - median failed: {e}")
        errors["median"] = None

    # k-medoid
    try:
        final, device = find_best_experimental(exp_dir, k_medoid, k)
        errors["k-medoid"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - k-medoid failed: {e}")
        errors["k-medoid"] = None

    # exponential decay with rssi field
    try:
        estimate_weights, device = assign_weights_rssi(exp_dir, k, alpha=alpha)
        pred = weighted_mean(estimate_weights)
        errors["exponential decay"] = distance(pred, device)
    except Exception as e:
        print(f"{exp_dir} - exponential decay failed: {e}")
        errors["exponential decay"] = None

    return errors

# experiment vs particular k or alpha
def plot_weighted_error_vs_experiment(experiments, k=4, alpha=0.7):
    errors = {}

    for exp_dir in experiments:
        try:
            estimate_weights, device = assign_weights_rssi(exp_dir, k=k, alpha=alpha)
            pred = weighted_mean(estimate_weights)
            err = distance(pred, device)
            print(f"{exp_dir}: Error = {err:.3f}")
            errors[exp_dir] = err
        except Exception as e:
            print(f"{exp_dir} failed: {e}")
            errors[exp_dir] = None

    # Prepare for plotting
    valid_exps = [exp for exp in errors if errors[exp] is not None]
    valid_errs = [errors[exp] for exp in valid_exps]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(valid_exps, valid_errs, color='darkorange')
    plt.xticks(rotation=45)
    plt.xlabel("Experiment")
    plt.ylabel("Error (Euclidean Distance)")
    plt.title(f"Exponential Decay Error per Experiment (k={k}, alpha={alpha})")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"graphset3/weighted_bar_k{k}_alpha{alpha}.png")
    plt.show()

# mean of all experiments over multiple k and particular alpha
def plot_mean_weighted_error_vs_k(experiments, k_range=range(4, 20), alpha=0.4):
    """
    Plots a bar graph of mean exponential decay error across all experiments for varying k values.

    Args:
        experiments (list): List of experiment names (e.g., ["exp_1", "exp_2", ...])
        k_range (iterable): Range of k values to try (e.g., range(4, 31))
        alpha (float): Alpha value for exponential decay weighting
    """
    mean_errors = {}

    for k in k_range:
        print(f"Running for k={k}")
        errors = []
        for exp_dir in experiments:
            try:
                estimate_weights, device = assign_weights_rssi(exp_dir, k=k, alpha=alpha)
                pred = weighted_mean(estimate_weights)
                err = distance(pred, device)
                errors.append(err)
            except Exception as e:
                print(f"{exp_dir} failed at k={k}: {e}")
                continue

        if errors:
            mean_errors[k] = np.mean(errors)
        else:
            mean_errors[k] = None

    # Filter valid ks
    valid_ks = [k for k in mean_errors if mean_errors[k] is not None]
    mean_vals = [mean_errors[k] for k in valid_ks]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(valid_ks, mean_vals, color='mediumseagreen')
    plt.xticks(valid_ks)
    plt.xlabel("k (Number of Points)")
    plt.ylabel("Mean Error (Euclidean Distance)")
    plt.title(f"Mean Exponential Decay Error vs k (alpha={alpha})")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"graphset3/mean_weighted_error_vs_k_alpha{alpha}.png")
    plt.show()


def plot_errors(experiments, error_dicts):
    """
    Plots the line graph comparing algorithm performance.
    """
    algorithms = list(error_dicts[0].keys())
    x = list(range(1, len(experiments) + 1))

    plt.figure(figsize=(12, 6))

    for algo in algorithms:
        y = [errs.get(algo, None) for errs in error_dicts]
        plt.plot(x, y, marker='o', label=algo)

    plt.xticks(x, experiments, rotation=45)
    plt.xlabel("Experiment")
    plt.ylabel("Error (Euclidean Distance)")
    plt.title("Comparison of Estimation Algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphset2/Exponential Decay")
    # plt.show()

def plot_errors_detailed(experiments, error_dicts, k, number):
    """
    Plots the line graph comparing algorithm performance.
    """
    algorithms = list(error_dicts[0].keys())
    x = list(range(1, len(experiments) + 1))

    plt.figure(figsize=(12, 6))

    for algo in algorithms:
        y = [errs.get(algo, None) for errs in error_dicts]
        plt.plot(x, y, marker='o', label=algo)

    plt.xticks(x, experiments, rotation=45)
    plt.xlabel("Experiment")
    plt.ylabel("Error (Euclidean Distance)")
    plt.title("Comparison of Estimation Algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphset3/Exponential_Decay{k}_{number}.png")
    # plt.show()

import numpy as np

def plot_mean_std_bar(error_dicts, k, alpha):
    """
    Plots bar graphs for mean and std deviation of errors across different methods.

    Args:
        error_dicts (list of dict): List of {algorithm: error} for each experiment
        k (int): Value of k used
        alpha (float): Alpha used in exponential decay
    """
    methods = list(error_dicts[0].keys())
    means = []
    stds = []

    for method in methods:
        values = [d.get(method) for d in error_dicts if d.get(method) is not None]
        means.append(np.mean(values))
        stds.append(np.std(values))

    x = np.arange(len(methods))

    # --- Plot Mean Error ---
    plt.figure(figsize=(10, 6))
    plt.bar(x, means, color='skyblue')
    plt.xticks(x, methods, rotation=45)
    plt.ylabel("Mean Error (Euclidean Distance)")
    plt.title(f"Mean Error per Method (k={k}, alpha={alpha})")
    plt.tight_layout()
    plt.savefig(f"graphset3/mean_error_k{k}_alpha{alpha}.png")
    # plt.show()

    # --- Plot Standard Deviation ---
    plt.figure(figsize=(10, 6))
    plt.bar(x, stds, color='salmon')
    plt.xticks(x, methods, rotation=45)
    plt.ylabel("Standard Deviation of Error")
    plt.title(f"Error Std Dev per Method (k={k}, alpha={alpha})")
    plt.tight_layout()
    plt.savefig(f"graphset3/std_error_k{k}_alpha{alpha}.png")
    # plt.show()



if __name__ == "__main__":
    experiments = [f"exp_{i}" for i in range(1, 37)]
    dc = {}
    # for k in [4,5]:
    #     for alph in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #         all_errors = []
    #         for exp in experiments:
    #             print(f"Running {exp}")
    #             result = run_all_algorithms(exp, k=k, alpha=alph, prev=None)
    #             all_errors.append(result)
    #         dc[(k, alph)] = all_errors
    #         # plot_just_weighted(exp)
    #         plot_errors_massive(experiments, all_errors, k, alph)
    # print(dc)


    all_errors = []
    k = 5
    for exp in experiments:
        print(f"Running {exp}")
        result = run_all_algorithms(exp, k=k, alpha=0.4, prev=None)
        all_errors.append(result)
    # plot_just_weighted(exp)
    plot_errors_detailed(experiments, all_errors, k, 0.4)
    plot_mean_std_bar(all_errors, k, 0.4)

    # with open("error_data", "w") as f:
    #     for key, val in dc.items():
    #         f.write(f"{key}: {val}\n")
    plot_mean_weighted_error_vs_k(experiments)
    # plot_weighted_error_vs_experiment(experiments, 6, 0.4)