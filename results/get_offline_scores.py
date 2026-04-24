import os
import pickle

import matplotlib.pyplot as plt

# Path to the pickle file you generated
FILE_PATH = (
    "/Users/batin13/Desktop/RL Projects/Experimental/offline_raw_scores_subset.pickle"
)


def read_and_inspect_data():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find the file at {FILE_PATH}")
        return

    # Load the pickle file
    with open(FILE_PATH, "rb") as handle:
        data = pickle.load(handle)

    print("--- Data Structure & Statistics ---")

    # Iterate through the nested dictionary
    for algorithm, datasets in data.items():
        for dataset, runs in datasets.items():
            # Skip empty datasets (since we filtered, many will be empty)
            if not runs:
                continue

            print(f"\nAlgorithm: {algorithm} | Dataset: {dataset}")
            print(f"Number of runs collected: {len(runs)}")

            # Plotting setup
            plt.figure(figsize=(10, 6))

            # Inspect each run and plot it
            for i, run_scores in enumerate(runs):
                print(f"  Run {i + 1}: Contains {len(run_scores)} evaluation steps.")

                # Plot the raw scores for this run
                plt.plot(run_scores, label=f"Run {i + 1}", alpha=0.8)

            # Format the plot
            plt.title(f"Raw Evaluation Scores: {algorithm} on {dataset}")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Raw Return")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # Show the plot
            plt.show()


if __name__ == "__main__":
    read_and_inspect_data()
