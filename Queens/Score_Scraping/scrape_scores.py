import re
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


from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re, time

def scroll_chat_to_top(driver, pause=1.0, max_scrolls=15):
    xpath = (
        "//div[contains(@class,'msg-s-message-list') "
        "and contains(@class,'msg-s-message-list--scroll-buffer') "
        "and contains(@class,'scrollable')]"
    )
    chat = driver.find_element(By.XPATH, xpath)
    last_h = driver.execute_script("return arguments[0].scrollHeight", chat)
    for _ in range(max_scrolls):
        driver.execute_script("arguments[0].scrollTop = 0", chat)
        time.sleep(pause)
        new_h = driver.execute_script("return arguments[0].scrollHeight", chat)
        if new_h == last_h:
            break
        last_h = new_h

def scrape_queens_via_bs(driver):
    scroll_chat_to_top(driver, pause=1.2, max_scrolls=20)

    # Grab the container’s HTML
    xpath = (
        "//div[contains(@class,'msg-s-message-list') "
        "and contains(@class,'msg-s-message-list--scroll-buffer') "
        "and contains(@class,'scrollable')]"
    )
    container = driver.find_element(By.XPATH, xpath)
    html = container.get_attribute("innerHTML")

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    lis = soup.select("li.msg-s-message-list__event")

     # Find the index of the real Today divider
    start = 0
    for i, li in enumerate(lis):
        t = li.find("time", class_="msg-s-message-list__time-heading")
        if t and t.get_text(strip=True) == "Today":
            start = i
            break

    # Walk from that <li> onward **including** the marker li itself
    results = []
    current_sender = "Unknown"
    for li in lis[start:]:            # <-- include the marker li
        # A) update sender if meta section present
        meta = li.select_one(".msg-s-message-group__meta")
        if meta:
            name_span = meta.select_one(".msg-s-message-group__name")
            if name_span and name_span.get_text(strip=True):
                current_sender = name_span.get_text(strip=True)

        # B) find bubble text (even in the marker li)
        bubble = li.select_one(".msg-s-event-listitem__body")
        if not bubble:
            continue
        text = bubble.get_text("\n", strip=True)

        # C) match first non-empty line
        first = next((line for line in text.splitlines() if line.strip()), "")
        if re.match(r"^Queens #\d{3}", first):
            results.append((current_sender, text))

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

        time.sleep(4)
        results = scrape_queens_via_bs(driver)
        return results
    except:
        print(f"Could not find button for {name}")


def main():
    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Anonymised"
    driver = initialise_driver(COOKIE_FILE)
    results = score_scraper(driver, name)
    print(results)

main()