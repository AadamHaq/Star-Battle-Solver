import re
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re
import time
import pickle
import os
from selenium.webdriver.common.keys import Keys


def load_cookies(driver, cookie_file):
    """
    Function: load_cookies retrieves cookies saved in the pickle file

    Args:
        driver: webdriver used (in this case Chrome driver)
        cookie_file: OS path to cookie pickle file
    
    Description: Retrieves cookies and loads them in the driver
    """
    if not os.path.exists(cookie_file):
        raise FileNotFoundError(f"Cookie file '{cookie_file}' not found. Please login manually and save cookies first by running get_cookies.py")
    
    with open(cookie_file, "rb") as f:
        cookies = pickle.load(f)
    for cookie in cookies:
        # Remove 'sameSite' if present — sometimes causes issues
        cookie.pop('sameSite', None)
        driver.add_cookie(cookie)


def initialise_driver(cookie_file):
    """
    Function: initialises the driver to be used

    Args:
        cookie_file: OS path to cookie pickle file
    
    Description: Initialises the driver, launches LinkedIn and loads the cookies
    """
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

    load_cookies(driver, cookie_file)
    print("Cookies loaded")

    return driver

def scrape_queens_messages_today(driver):
    today_label = datetime.now().strftime("%b %d")  # e.g., "May 13"
    results = []
    is_today = False

    elements = driver.find_elements(By.XPATH, "//*")

    for el in elements:
        # Detect today's date header
        if el.tag_name == "time" and "msg-s-message-list__time-heading" in el.get_attribute("class"):
            is_today = el.text.strip() == today_label
            continue

        if not is_today:
            continue

        if "msg-s-event-listitem" in el.get_attribute("class"):
            try:
                # Extract message text
                msg_el = el.find_element(By.CLASS_NAME, "msg-s-event-listitem__body")
                message = msg_el.text.strip()

                # Match pattern like "Queens #123"
                if re.match(r"^Queens\s+#\d{3}", message):
                    name_el = el.find_element(By.CLASS_NAME, "msg-s-message-group__name")
                    sender = name_el.text.strip()

                    results.append({
                        "sender": sender,
                        "message": message
                    })
            except Exception:
                continue

    return results





def score_scraper(driver, name):

    driver.get("https://www.linkedin.com/feed/") # Open normal linkedin first to confirm/set cookies
    time.sleep(8)

    messaging_icon = driver.find_element(By.XPATH, "//li-icon[@type='nav-small-messaging-icon']")
    messaging_icon.click()

    time.sleep(5)
    search_bar = driver.find_element(By.ID, "search-conversations")
    search_bar.send_keys(name + Keys.ENTER)

    try:
        time.sleep(5)
        conversation = driver.find_element(By.XPATH, f"//span[text()='{name}']/ancestor::div[contains(@class, 'msg-conversation-card__content--selectable')]")
        conversation.click()

        time.sleep(5)
        results = scrape_queens_messages_today(driver)
        return results
    except:
        print(f"Could not find button for {name}")


def main():
    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Anonymised Name"
    driver = initialise_driver(COOKIE_FILE)
    results = score_scraper(driver, name)
    print(results)

main()