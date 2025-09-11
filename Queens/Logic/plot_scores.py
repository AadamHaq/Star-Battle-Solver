from dotenv import load_dotenv
import os
from pathlib import Path
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env") 


def name_to_env_key(name):
    cleaned = name.replace('.', '').strip()
    return "ALIAS_" + cleaned.replace(" ", "_").upper()

def get_dow(day):
    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    return dow_names[((day + 2) % 7)]  # Monday is day % 7 == 6 → index 6 = "Sat"

# Build alias dict directly from os.environ
alias_dict = {
    key: value for key, value in os.environ.items() if key.startswith("ALIAS_")
    }


def prepare_csv(results):
    # Extract all Queens numbers from results
    day_numbers = []
    for _, text in results:
        match = re.search(r'Queens\s+#(\d+)', text)
        if match:
            day_numbers.append(int(match.group(1)))

    if not day_numbers:
        raise ValueError("No Queens numbers found in results.")

    # Use the highest day number (latest puzzle)
    latest_day = max(day_numbers)
    dow = get_dow(latest_day)

    row = {"Day": latest_day, "DoW": dow}

    for name, text in results:
        if f'Queens #{latest_day}' not in text:
            continue  # Skip if it's not from the latest puzzle

        alias_key = name_to_env_key(name)
        alias = alias_dict.get(alias_key)

        print(f"Processing name '{name}' → alias key '{alias_key}' → alias value '{alias}'")

        if alias:
            score_match = re.search(r'(\d{1,2}:\d{2})', text)
            if score_match:
                row[alias] = score_match.group(1)

    print(f"Row obtained: {row}")
    return row

def backfill_previous_day(results, latest_day):
    previous_day = latest_day - 1
    csv_file = project_root / "scores.csv"

    if not csv_file.exists():
        return  # Nothing to backfill

    df = pd.read_csv(csv_file)
    if previous_day not in df["Day"].values:
        return  # No row to update

    prev_idx = df.index[df["Day"] == previous_day][0]
    modified = False

    for name, text in results:
        if f'Queens #{previous_day}' not in text:
            continue

        alias_key = name_to_env_key(name)
        alias = alias_dict.get(alias_key)

        if alias and alias in df.columns and pd.isna(df.at[prev_idx, alias]):
            score_match = re.search(r'(\d{1,2}:\d{2})', text)
            if score_match:
                df.at[prev_idx, alias] = score_match.group(1)
                print(f"[Backfill] Inserted {alias}'s score into Day {previous_day}")
                modified = True

    if modified:
        df.to_csv(csv_file, index=False)
        print(f"[Backfill] Updated Day {previous_day} in {csv_file.name}")

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

    backfill_previous_day(results, row["Day"])

def plot_graph():
    # Load CSV
    df = pd.read_csv(project_root / "scores.csv")

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

    ax = plt.gca()

    # Set x-axis labels with Day and DoW
    x_ticks = df_seconds["Day"]
    x_labels = [f"{day}\n{dow}" for day, dow in zip(df_seconds["Day"], df_seconds["DoW"])]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Set y-axis limit to 5 minutes
    ax.set_ylim(0, 360)

    # Format y-axis ticks to mm:ss
    def format_mmss(x, _):
        if x < 0 or pd.isna(x):
            return ''
        m, s = divmod(int(x), 60)
        return f"{m}:{s:02}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_mmss))

    # Labels and styling
    plt.xlabel("Day")
    plt.ylabel("Time (mm:ss)")
    plt.title("Queens Scores Over Time")
    plt.legend(title="Player")
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig("queens_scores_plot.png")


def main():
    plot_graph()

if __name__ == "__main__":
    main()
