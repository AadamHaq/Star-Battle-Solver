from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time

def share_score(driver, name):
    """
    Function: Shares Queens score to a group chat

    Args:
        driver: Selenium driver that was initialised
        name: Name of group chat that the score will be sent to

    Description: Removes pop up for streak freeze, scrolls down and shares score
    to a group chat name entered.
    """
    try: # Streak freeze pop up problem. May need to change button path here when it comes back again
        got_it_button = driver.find_element(By.XPATH, "//button[text()='Got it']")
        got_it_button.click()
        time.sleep(0.5)
    except:
        pass  # No pop-up, continue normally

    # Click on page at crown photo (not a hyperlink) so scrolling works
    image_section = driver.find_element(By.CSS_SELECTOR, ".pr-top__image")
    ActionChains(driver).move_to_element(image_section).click().perform()
    time.sleep(0.5)

    # 3 DOWN arrow keys to scroll the page or reveal elements
    body = driver.find_element(By.TAG_NAME, "body")
    for _ in range(3):
        body.send_keys(Keys.DOWN)
        time.sleep(0.3)

    print("Clicking share button")
    send_button = driver.find_element(
    By.XPATH,
    "//span[normalize-space()='Send']/preceding-sibling::button" # Used 'Send' text instead of button due to errors
    )
    send_button.click()

    time.sleep(0.5)

    try:
        send_to = driver.find_element(
            By.XPATH,
            f"//div[@class='msg-connections-typeahead__entity-description']"
            f"[.//dt[contains(normalize-space(), '{name}')]]" # Name of person to send to and click
            "/ancestor::button"
        )
        send_to.click()
    except:
        input_box = driver.find_element(By.XPATH, "//input[@placeholder='Type a name']") # Type group chat name
        input_box.send_keys(name + Keys.ENTER)

        time.sleep(1)

        send_to = driver.find_element(
            By.XPATH,
            f"//div[@class='msg-connections-typeahead__entity-description']"
            f"[.//dt[contains(normalize-space(), '{name}')]]" # Name of person to send to and click
            "/ancestor::button"
        )
        send_to.click()
    try:
        time.sleep(0.5)
        textbox = driver.find_element(
            By.XPATH,
            "//div[contains(@class, 'msg-form__contenteditable') and @contenteditable='true']" # Text box for message
        )
        textbox.click()

        # Tab to get to the send button. This was the easiest way as clicking the send button would not work
        actions = ActionChains(driver)
        actions.send_keys(Keys.TAB).pause(0.2).send_keys(Keys.ENTER).perform()

        print("Tab + Enter sent successfully.")
        time.sleep(2)

    except:
        print("Could not find user to send to")
