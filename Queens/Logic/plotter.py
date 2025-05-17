from dotenv import load_dotenv
import os
from pathlib import Path
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env") 


def name_to_env_key(name):
    cleaned = name.replace('.', '').strip()
    return "ALIAS_" + cleaned.replace(" ", "_").upper()

def get_dow(day):
    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    return dow_names[(day % 7) + 2]  # Monday is day % 7 == 6 → index 6 = "Sat"

# Build alias dict directly from os.environ
alias_dict = {
    key: value for key, value in os.environ.items() if key.startswith("ALIAS_")
    }


def prepare_csv(results):
    day_match = re.search(r'Queens\s+#(\d+)', results[0][1])
    day = int(day_match.group(1)) if day_match else None

    dow = get_dow(day)

    row = {"Day": day, "DoW": dow}
    for name, text in results:
        alias_key = name_to_env_key(name)
        alias = alias_dict.get(alias_key)

        print(f"Processing name '{name}' → alias key '{alias_key}' → alias value '{alias}'")

        if alias:
            score_match = re.search(r'(\d{1,2}:\d{2})', text)
            if score_match:
                row[alias] = score_match.group(1)

    print(f"Row obtained: {row}")
    return row

def write_csv(results):
    row = prepare_csv(results)

    # Path to Queens/scores.csv
    csv_file = project_root / "scores.csv"
    file_exists = csv_file.exists()

    # Define the column order explicitly:
    columns = ["Day", "DoW", "Me", "HK", "AH", "AS", "JM", "ZK", "Backtracking"]

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        # Ensure all columns are present in row dict, fill missing with empty string
        row_filled = {col: row.get(col, '') for col in columns}

        writer.writerow(row_filled)

    print(f"Row written to {csv_file.name}: {row}")

def plot_graph():
    # Load CSV
    df = pd.read_csv("Queens/scores.csv")

    # Convert mm:ss to total seconds
    def time_to_seconds(t):
        if pd.isna(t):
            return None
        try:
            m, s = map(int, t.split(":"))
            return m * 60 + s
        except:
            return None

    # Convert to seconds for plotting
    df_seconds = df.copy()
    for col in df.columns:
        if col not in ["Day", "DoW"]:
            df_seconds[col] = df[col].apply(time_to_seconds)

    # Plot
    plt.figure(figsize=(10, 6))
    for col in df_seconds.columns:
        if col not in ["Day", "DoW"]:
            plt.plot(df_seconds["Day"], df_seconds[col], marker='o', label=col)

    # Set x-axis labels with Day and DoW
    ax = plt.gca()
    x_ticks = df_seconds["Day"]
    x_labels = [f"{day}\n{dow}" for day, dow in zip(df_seconds["Day"], df_seconds["DoW"])]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Format y-ticks manually to mm:ss
    y_ticks = ax.get_yticks()
    y_labels = []
    for val in y_ticks:
        if pd.isna(val) or val < 0:
            y_labels.append('')
        else:
            m, s = divmod(int(val), 60)
            y_labels.append(f"{m}:{s:02}")
    ax.set_yticklabels(y_labels)

    # Labels and styling
    plt.xlabel("Day")
    plt.ylabel("Time (mm:ss)")
    plt.title("Queens Scores Over Time")
    plt.legend(title="Player")
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig("queens_scores_plot.png")
    plt.show()


def main():
    plot_graph()

if __name__ == "__main__":
    main()
