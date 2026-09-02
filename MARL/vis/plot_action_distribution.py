import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping from integer action codes to descriptive strings
ACTIONS_MAP = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

def load_actions(file_path: str) -> pd.Series:
    """Load the actions stored in a NumPy ``.npy`` file.

    The file is expected to contain a one‑dimensional array of integers.
    The returned Series contains the corresponding string labels.
    """
    # Load raw integer array
    raw = np.load(file_path, allow_pickle=True)
    # Ensure we have a 1‑D array
    raw = np.ravel(raw)
    # Convert to pandas Series for convenient counting
    s = pd.Series(raw, name="action")
    # Map integer codes to descriptive strings
    s = s.map(ACTIONS_MAP)
    return s

def plot_distribution(actions: pd.Series, output_path: str = "action_distribution.png"):
    """Create a bar plot showing the frequency of each action.

    Parameters
    ----------
    actions: pd.Series
        Series of action strings.
    output_path: str, optional
        Filename for the saved figure. Defaults to ``action_distribution.png``.
    """
    # Count occurrences, ensuring the order follows the mapping definition
    counts = actions.value_counts().reindex(list(ACTIONS_MAP.values()))
    # Replace possible NaN (if an action never appears) with 0
    counts = counts.fillna(0).astype(int)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette="muted")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Actions")
    # Annotate counts on top of bars for clarity
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Assume the .npy file is located in the same directory as this script
    import pathlib
    script_dir = pathlib.Path(__file__).parent
    # actions_file = script_dir / "actions.npy"
    actions_file = "actions.npy"
    # if not actions_file.is_file():
    #     raise FileNotFoundError(f"Could not find {actions_file}")
    actions_series = load_actions(str(actions_file))
    plot_distribution(actions_series, script_dir / "action_distribution.png")
    print(f"Plot saved to {script_dir / 'action_distribution.png'}")
