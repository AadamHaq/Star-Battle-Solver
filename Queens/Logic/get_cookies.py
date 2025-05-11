from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import pickle
import time

options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.linkedin.com/login")

# One minute delay to log in manually
print("Please log in manually. Waiting 60 seconds...")
time.sleep(60)  

# Save cookies after login as pickle file
cookies = driver.get_cookies()
with open("linkedin_cookies.pkl", "wb") as f:
    pickle.dump(cookies, f)

print("Cookies saved.")
driver.quit()
