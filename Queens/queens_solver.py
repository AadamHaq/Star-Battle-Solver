from Logic.scraper import initialise_driver, scraper
from Logic.Solvers.naive_backtracking import backtracking
from Logic.inputter import input_solution
import time

def main(cookie_file):
    """
    Function: main runs all other functions for the game

    Args:
        cookie_file: .pkl file that can be retrieved by running get_cookies.py
    
    Description: Runs all other functions into one seamless solution then quits the driver

    Returns: None
    """

    driver = initialise_driver(cookie_file)

    data = scraper(driver)

    board = data['board']
    N = data['board_size']
    print(board)
    solution = backtracking(board, N)

    print(solution)
    solution_1_indexed = [(r + 1, c + 1) for r, c in sorted(solution)] # Sorted purely for visual purposes
    input_solution(driver, solution_1_indexed)

    time.sleep(10)

    driver.quit()

if __name__ == "__main__":
    COOKIE_FILE = "linkedin_cookies.pkl"
    main(COOKIE_FILE)