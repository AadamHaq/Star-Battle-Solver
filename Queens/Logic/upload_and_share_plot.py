from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

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
    # 4. Press TAB then ENTER to send
    actions = ActionChains(driver)

    # Repeat TAB 6 times with a 0.2s pause between each
    for _ in range(6):
        actions.send_keys(Keys.TAB).pause(0.2)

    # Finally, press ENTER
    actions.send_keys(Keys.ENTER).perform()
    print("✅ Image sent via Tab + Enter.")
    time.sleep(5)
