from selenium.webdriver.common.by import By
import time

def share_score(driver, name):

    share_button = driver.find_element(By.XPATH, "//button[.//svg/use[contains(@href, '#send-privately-medium')]]") # id changes each time
    share_button.click() # share button

    time.sleep(1)

    input_box = driver.find_element(By.XPATH, "//input[@placeholder='Type a name']")
    input_box.send_keys(name)

    time.sleep(1)

    try:
        send_to = driver.find_element( # id changes for each person/group so had to do this way
        By.XPATH, f"//button[contains(@class, 'msg-connections-typeahead__search-result')]//*[contains(text(), '{name}')]/ancestor::button" 
        )
        send_to.click()

        submit = driver.find_element(By.XPATH, "//button[text()='Send']")
        submit.click()

    except:
        print("Could not find user to send to")

    
