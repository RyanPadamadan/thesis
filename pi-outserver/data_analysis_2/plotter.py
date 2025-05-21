import matplotlib.pyplot as plt
from algorithms import (
    find_best_experimental,
    k_means,
    k_means_ewma,
    k_medoid,
    median,
    weighted_mean
)
from spatial_manager import assign_weights
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
    try:
        final, device = find_best_experimental(exp_dir, k_means_ewma, k, it=True, prev=prev)
        errors["k-means EWMA"] = distance(final, device)
    except Exception as e:
        print(f"{exp_dir} - k-means EWMA failed: {e}")
        errors["k-means EWMA"] = None

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

    # exponential decay (from surface proximity)
    try:
        estimate_weights, device = assign_weights(exp_dir)
        pred = weighted_mean(estimate_weights)
        errors["exponential decay"] = distance(pred, device)
    except Exception as e:
        print(f"{exp_dir} - exponential decay failed: {e}")
        errors["exponential decay"] = None

    return errors


def plot_just_weighted(exp_dir, k=4):
    """
    Runs exponential decay (weighted mean) with varying P values from 1 to 4.
    Plots error for each P.
    """
    errors = {}
    Ps = [1, 2, 3, 4]

    for P in Ps:
        try:
            estimate_weights, device = assign_weights(exp_dir, P=P)
            pred = weighted_mean(estimate_weights)
            err = distance(pred, device)
            print(f"P={P}: Error = {err:.3f}")
            errors[P] = err
        except Exception as e:
            print(f"{exp_dir} - P={P} failed: {e}")
            errors[P] = None

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(list(errors.keys()), list(errors.values()), marker='o', color='purple', label='Exponential Decay')
    plt.title(f"Error vs P for Weighted Mean - {exp_dir}")
    plt.xlabel("P value")
    plt.ylabel("Error (Euclidean Distance)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"weighted_error_{exp_dir}.png")
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
    plt.savefig("Exponential Decay 1")
    plt.show()


if __name__ == "__main__":
    experiments = [f"exp_{i}" for i in range(1, 8)]
    all_errors = []

    for exp in experiments:
        print(f"Running {exp}")
        # result = run_all_algorithms(exp, k=4, alpha=0.7, prev=None)
        # all_errors.append(result)

        plot_just_weighted(exp)

    plot_errors(experiments, all_errors)
