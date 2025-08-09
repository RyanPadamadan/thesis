import matplotlib.pyplot as plt
from algorithms import (
    find_best_experimental,
    k_means,
    k_means_ewma,
    k_medoid,
    median,
    weighted_mean
)

from algorithms import *
from spatial_manager import assign_weights, assign_weights_rssi, assign_weights_rssi_v2
from data_cleaner import *
import numpy as np

import csv
import os
import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_all_algorithms(exp_dir, k=4, alpha=0.7, prev=None, Q=3):
    """
    Runs each algorithm on the given experiment and returns their errors,
    using the SAME set of estimates for all clustering-based methods.

    Returns:
        dict: {algorithm_name: error}
    """
    errors = {}

    # ---- Precompute once and reuse ----
    try:
        points, device = setup_experiment_no_mesh(exp_dir)
        estimates = get_estimates(points, k)
    except Exception as e:
        # If we can't even build estimates, everything except exp-decay fails
        print(f"{exp_dir} - estimate generation failed: {e}")
        estimates, device = None, None

    # k-means
    try:
        if estimates is None:
            raise RuntimeError("No estimates available")
        final = k_means(estimates)
        errors["k-means"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - k-means failed: {e}")
        errors["k-means"] = None

    # median
    try:
        if estimates is None:
            raise RuntimeError("No estimates available")
        final = median(estimates)
        errors["median"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - median failed: {e}")
        errors["median"] = None

    # k-medoid
    try:
        if estimates is None:
            raise RuntimeError("No estimates available")
        final = k_medoid(estimates)
        errors["k-medoid"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - k-medoid failed: {e}")
        errors["k-medoid"] = None

    # exponential decay with RSSI field
    # Prefer a version that can reuse the SAME 'estimates' if available.
    try:
        if estimates is None:
            raise RuntimeError("No estimates available")
        estimate_weights, device2 = assign_weights_rssi_v2(
            exp_dir,
            estimates,
            Q=Q
        )
        pred = weighted_mean(estimate_weights)
        # device2 should match `device`; keep device2 in case downstream differs
        errors["exponential decay"] = distance(pred, device2)
    except Exception as e:
        print(f"{exp_dir} - exponential decay failed: {e}")
        errors["exponential decay"] = None

    return errors




def plot_alpha_vs_mean_error(experiments, k=4, alphas=None, output_dir="graphset_alpha_summary"):
    if alphas is None:
        alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  # 0.1 to 1.0

    means = []
    stds = []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open CSV to store data
    csv_path = os.path.join(output_dir, f"alpha_vs_error_k{k}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Alpha", "Mean Error", "Std Deviation"])

        for alpha in alphas:
            errors = []

            for exp_dir in experiments:
                try:
                    estimate_weights, device = assign_weights_rssi(exp_dir, k=k, alpha=alpha)
                    pred = weighted_mean(estimate_weights)
                    err = distance(pred, device)
                    errors.append(err)
                except Exception as e:
                    print(f"{exp_dir} failed at alpha={alpha}: {e}")

            if errors:
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                print(f"alpha={alpha:.2f}: mean={mean_err:.3f}, std={std_err:.3f}")
                means.append(mean_err)
                stds.append(std_err)
                writer.writerow([alpha, mean_err, std_err])
            else:
                means.append(None)
                stds.append(None)
                writer.writerow([alpha, "NaN", "NaN"])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, means, yerr=stds, fmt='-o', color='darkblue', ecolor='lightgray', capsize=5)
    plt.xlabel("Alpha")
    plt.ylabel("Mean Localization Error")
    plt.title(f"Mean Error vs Alpha (k={k})")
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"alpha_vs_mean_error_k{k}.png")
    # plt.savefig(plot_path)
    plt.show()
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary data to {csv_path}")


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
    csv_path = f"graphset3/mean_weighted_error_vs_k_alpha{alpha}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["k", "Mean Error"])
        for k in valid_ks:
            writer.writerow([k, mean_errors[k]])
    print(f"Saving csv: {csv_path}")

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
    Saves experiment errors to CSV and plots the comparison graph.
    Skips experiments where there are no usable results.
    """
    # Keep only experiments that have at least one non-None error dict value
    paired = []
    for exp_name, errs in zip(experiments, error_dicts):
        if errs is None:
            continue
        if any(v is not None for v in errs.values()):
            paired.append((exp_name, errs))

    if not paired:
        print("No experiments with usable results to plot.")
        return

    experiments_f, error_dicts_f = zip(*paired)
    experiments_f, error_dicts_f = list(experiments_f), list(error_dicts_f)

    algorithms = list(error_dicts_f[0].keys())
    x = list(range(1, len(experiments_f) + 1))

    # Prepare CSV data
    csv_data = []
    for exp_name, errs in zip(experiments_f, error_dicts_f):
        row = {"exp_no": exp_name}
        for algo in algorithms:
            row[algo] = errs.get(algo, None)
        csv_data.append(row)

    # Ensure folder exists
    import os
    os.makedirs("graphset3", exist_ok=True)

    # Save to CSV
    pname = f"graphset3/errors_k{k}_{number}.csv"
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv(pname, index=False)
    print(f"Saved csv at {pname}")

    # Plot (matplotlib will leave gaps for None values)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        y = [errs.get(algo, None) for errs in error_dicts_f]
        plt.plot(x, y, marker='o', label=algo)

    plt.xticks(x, experiments_f, rotation=45)
    plt.xlabel("Experiment")
    plt.ylabel("Error (Euclidean Distance)")
    plt.title("Comparison of Estimation Algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphset3/Exponential_Decay{k}_{number}.png")
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_mean_std_bar(error_dicts, k, alpha, output_dir="exps2_final_barplots"):
    """
    Plots a bar graph for mean errors with standard deviation as error bars,
    and saves the values to a CSV.
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    methods = list(error_dicts[0].keys())
    means = []
    stds = []

    for method in methods:
        values = [d.get(method) for d in error_dicts if d.get(method) is not None]
        means.append(np.mean(values))
        stds.append(np.std(values))

    x = np.arange(len(methods))

    # --- Plot Mean Error with Std Dev Error Bars ---
    plt.figure(figsize=(9, 5))
    bar_width = 0.6
    plt.bar(x, means, yerr=stds, width=bar_width, capsize=4, color='skyblue', ecolor='black')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylabel("Mean Error (Euclidean Distance)")

    # Dynamic Y-axis limit (5% headroom above top error bar)
    upper_limit = max(m + s for m, s in zip(means, stds)) * 1.05
    lower_limit = min(0, min(m - s for m, s in zip(means, stds)))  # allow for error bars below 0
    plt.ylim(lower_limit, upper_limit)

    plt.title(f"Mean Error per Method with Std Dev (k={k}, alpha={alpha})")
    plt.tight_layout()
    mean_plot_path = os.path.join(output_dir, f"mean_error_with_std_k{k}_alpha{alpha}.png")
    plt.savefig(mean_plot_path)
    plt.show()

    # --- Save Mean and Std to CSV ---
    csv_path = os.path.join(output_dir, f"summary_stats_k{k}_alpha{alpha}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Mean Error", "Standard Deviation"])
        for method, mean, std in zip(methods, means, stds):
            writer.writerow([method, mean, std])

    print(f"Saved mean error plot with std dev to {mean_plot_path}")
    print(f"Saved summary statistics to {csv_path}")

if __name__ == "__main__":
    # to_skip = [8,9,10,18, 20, 21, 27, 28, 29, 30, 31, 32]
    experiments = []
    for i in range(1,37):
        # if i not in to_skip:
        experiments.append(f"exp_{i}")
    # dc = {}
    # for k in [4,5]:
    #     for alph in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #         all_errors = []
    #         for exp in experiments:
    #             print(f"Running {exp}")
    #             result = run_all_algorithms(exp, k=k, alpha=alph, prev=None)
    #             all_errors.append(result)
    #         dc[(k, alph)] = all_errors
            # plot_just_weighted(exp)
            # plot_errors_detailed(experiments, all_errors, k, alph)
    # print(dc)


    all_errors = []
    k = 4
    for exp in experiments:
        print(f"Running {exp}")
        result = run_all_algorithms(exp, k=k, alpha=0.8, prev=None)
        all_errors.append(result)
    plot_errors_detailed(experiments, all_errors, k, 0.4)
    plot_mean_std_bar(all_errors, k, 0.4)
    # plot_mean_weighted_error_vs_k(experiments)
    # plot_alpha_vs_mean_error(experiments)