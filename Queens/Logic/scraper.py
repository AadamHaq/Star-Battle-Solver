from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re
import time
import pickle
import os

COOKIE_FILE = "linkedin_cookies.pkl" # Use get_cookies to retrieve pickle file the first time

def load_cookies(driver, cookie_file):
    if not os.path.exists(cookie_file):
        raise FileNotFoundError(f"Cookie file '{cookie_file}' not found. Please login manually and save cookies first by running get_cookies.py")
    
    with open(cookie_file, "rb") as f:
        cookies = pickle.load(f)
    for cookie in cookies:
        # Remove 'sameSite' if present — sometimes causes issues
        cookie.pop('sameSite', None)
        driver.add_cookie(cookie)


def scrape_queens_board_metadata():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # Suppress console noise
    chrome_options.add_argument("--log-level=3")  # ERROR level only
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])  # Hide DevTools logs
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,  # Disable browser notifications
        "profile.default_content_setting_values.media_stream_mic": 2,
        "profile.default_content_setting_values.media_stream_camera": 2,
        "profile.default_content_setting_values.geolocation": 2,
    })

    driver.get("https://www.linkedin.com") # Open normal linkedin first to confirm/set cookies
    time.sleep(3)

    load_cookies(driver, COOKIE_FILE)

    driver.get("https://www.linkedin.com/games/queens") 
    time.sleep(5)

    board = driver.find_element(By.ID, "queens-grid")

    style = board.get_attribute("style")
    rows = int(re.search(r"--rows:\s*(\d+)", style).group(1))
    cols = int(re.search(r"--cols:\s*(\d+)", style).group(1))
    assert rows == cols, "Grid should be square"

    size = rows
    board_matrix = [[None for _ in range(size)] for _ in range(size)]
    color_region_matrix = [[None for _ in range(size)] for _ in range(size)]

    cells = board.find_elements(By.CLASS_NAME, "queens-cell-with-border")

    for cell in cells:
        class_attr = cell.get_attribute("class")
        aria = cell.get_attribute("aria-label")
        
        color_match = re.search(r"cell-color-(\d+)", class_attr)
        color_index = int(color_match.group(1)) if color_match else None

        aria_match = re.search(r"row (\d+), column (\d+)", aria)
        if not aria_match:
            continue

        row = int(aria_match.group(1)) - 1  # 1-indexed to 0-indexed
        col = int(aria_match.group(2)) - 1

        color_region_matrix[row][col] = color_index

    driver.quit()

    return {
        "board_size": size,
        "color_regions": color_region_matrix
    }

if __name__ == "__main__":
    data = scrape_queens_board_metadata()
    for row in data["color_regions"]:
        print(row)
