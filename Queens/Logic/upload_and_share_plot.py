from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd
import random
import numpy as np
import os
from dotenv import load_dotenv

def time_to_seconds(t):
    """Convert mm:ss string to seconds (or NaN if missing)."""
    if pd.isna(t) or t == "":
        return np.nan
    try:
        mins, secs = t.split(":")
        return int(mins) * 60 + int(secs)
    except:
        return np.nan


def get_fun_fact(df):
    # --- Load aliases from .env ---
    load_dotenv()
    alias_map = {}
    for key, value in os.environ.items():
        if key.startswith("ALIAS_"):
            name_parts = key.replace("ALIAS_", "").split("_")
            first_name = name_parts[0].capitalize()  # only first name
            alias_map[value] = first_name

    players = [p for p in df.columns if p not in ["Day", "DoW", "Backtracking"]]

    # Convert all times to seconds
    df_secs = df.copy()
    for p in players:
        df_secs[p] = df_secs[p].apply(time_to_seconds)

    # Use most recent row as "today"
    today = df_secs.iloc[-1]
    last14 = df_secs.iloc[-15:-1]  # last 14 days before today

    fun_facts = []

    # --- Fastest solve today ---
    today_valid = today[players].dropna()
    if not today_valid.empty:
        fastest_player = today_valid.idxmin()
        fastest_time = today_valid.min()
        name = alias_map.get(fastest_player, fastest_player)
        fun_facts.append(f"🏆 Fastest today: {name} with {int(fastest_time//60)}:{int(fastest_time%60):02d}")

    # --- Bad day check (> 5:00) ---
    if not today_valid.empty:
        bad_day = today_valid[today_valid > 300]  # 300 seconds = 5 mins
        if not bad_day.empty:
            # Get all names using alias_map
            names = [alias_map.get(player, player) for player in bad_day.index]
            if len(names) == 1:
                name_str = names[0]
            else:
                name_str = ", ".join(names[:-1]) + " and " + names[-1]
            # Use the maximum time among the bad players for display
            max_secs = bad_day.max()
            fun_facts.append(
                f"😬 {name_str} had a bad day with up to {int(max_secs//60)}:{int(max_secs%60):02d}"
            )

    # --- Most improved (today vs average of last 14 days) ---
    if not today_valid.empty:
        improvements = {}
        for p in players:
            if not pd.isna(today[p]):
                past_avg = last14[p].dropna().mean()
                if past_avg and past_avg > 0:
                    diff = past_avg - today[p]
                    improvements[p] = diff
        if improvements:
            most_improved = max(improvements, key=improvements.get)
            gain = improvements[most_improved]
            if gain > 0:
                name = alias_map.get(most_improved, most_improved)
                fun_facts.append(f"📈 Most improved: {name}, {int(gain)}s faster than his 2-week average!")

    # --- Best recent average (last 14 days) ---
    averages = {p: last14[p].dropna().mean() for p in players}
    averages = {p: v for p, v in averages.items() if v and not np.isnan(v)}
    if averages:
        best_avg_player = min(averages, key=averages.get)
        best_avg_time = averages[best_avg_player]
        name = alias_map.get(best_avg_player, best_avg_player)
        fun_facts.append(f"🔥 Best 2-week average: {name} with {int(best_avg_time//60)}:{int(best_avg_time%60):02d}")

    # --- Current win streaks (consecutive days with valid times) ---
    winners = df_secs[players].idxmin(axis=1)

    # Track current consecutive win streaks for each player
    current_streaks = {p: 0 for p in players}
    for p in players:
        streak = 0
        # iterate backwards through days
        for w in reversed(winners):
            if w == p:
                streak += 1
            else:
                break
        current_streaks[p] = streak

    # Find player with the current longest win streak
    best_player = max(current_streaks, key=current_streaks.get)
    if current_streaks[best_player] > 1:  # only count streaks > 1
        name = alias_map.get(best_player, best_player)
        fun_facts.append(f"🔥 {name} is on a {current_streaks[best_player]}-day win streak!")

    # --- Big drop check (yesterday 1st → today last) ---
    if len(df_secs) >= 2:
        yesterday = df_secs.iloc[-2][players].dropna()
        if not yesterday.empty and not today_valid.empty:
            yesterday_winner = yesterday.idxmin()
            today_loser = today_valid.idxmax()
            if yesterday_winner == today_loser:
                name = alias_map.get(yesterday_winner, yesterday_winner)
                fun_facts.append(f"⬇️ {name} went from 1st yesterday to last today!")

    # --- Pick one fact to show ---
    if fun_facts:
        print(fun_facts)
        return random.choice(fun_facts)
    else:
        return "🤔 No valid stats available for today!"


# Path to the plot image
image_path = r"C:\Users\user\OneDrive\Documents\GitHub\Star-Battle-Solver\queens_scores_plot.png"
def upload_plot(driver):
    # 1. Find the image-specific input (accepts image/*)
    upload_input = driver.find_element(By.XPATH, "//input[@type='file' and contains(@accept, 'image')]")

    # 2. Upload the file
    upload_input.send_keys(image_path)
    print("📤 Image file sent to upload input.")
    
    try:
        # Option A: check for generic file preview (not just images)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'msg-attachment-preview')]"))
        )
        print("Option A")
    except:
        # Option B: wait for a "Send" button to become active again
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'msg-form__send-button')]"))
        )
        print("Option B")
    print("🖼️ Image preview appeared in chat.")

    time.sleep(3)

    textbox = driver.find_element(
        By.XPATH,
        "//div[contains(@class, 'msg-form__contenteditable') and @contenteditable='true']"
    )
    textbox.click()
    time.sleep(0.2)

    df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\GitHub\Star-Battle-Solver\Queens\scores.csv")
    fun_fact = get_fun_fact(df)

    actions = ActionChains(driver)
    actions.send_keys(fun_fact).perform()
    time.sleep(0.5)

    # 4. Press TAB then ENTER to send
    actions = ActionChains(driver)

    # Repeat TAB 6 times with a 0.2s pause between each
    for _ in range(6):
        actions.send_keys(Keys.TAB).pause(0.2)

    # Finally, press ENTER
    actions.send_keys(Keys.ENTER).perform()
    print("✅ Image sent via Tab + Enter.")
    time.sleep(5)

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\GitHub\Star-Battle-Solver\Queens\scores.csv")
    test = get_fun_fact(df)
    print(test)