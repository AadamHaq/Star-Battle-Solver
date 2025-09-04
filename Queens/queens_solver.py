from Logic.game_scraper import initialise_driver, scraper
from Logic.Solvers.naive_backtracking import backtracking
from Logic.game_inputter import input_solution
from Logic.share_score_after_play import share_score
from Logic.computer_vision import get_image, computer_vision
import time

def main(cookie_file, name):
    """
    Function: main runs all other functions for the game

    Args:
        cookie_file: .pkl file that can be retrieved by running get_cookies.py
        name: Name of group chat that the score will be sent to
    
    Description: Runs all other functions into one seamless solution then quits the driver

    Returns: None
    """

    driver = initialise_driver(cookie_file)

    try:
        path = "queens_board.png"
        get_image(driver, path)
        board = computer_vision(path)
        N = len(board)
        print(board)

    except:
        data = scraper(driver)

        board = data['board']
        N = data['board_size']
        print(board)
    solution = backtracking(board, N)

    print(solution)
    solution_1_indexed = [(r + 1, c + 1) for r, c in solution]
    input_solution(driver, solution_1_indexed)

    time.sleep(5)

    # Inject auto-clean for LinkedIn modals
    driver.execute_script("""
        const observer = new MutationObserver(() => {
            document.querySelectorAll(".artdeco-modal[role='dialog']")
                .forEach(el => el.remove());
        });
        observer.observe(document.body, { childList: true, subtree: true });
    """)

    time.sleep(1)
    share_score(driver, name)

    driver.quit()

if __name__ == "__main__":
    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Queens + Zip Daily"
    main(COOKIE_FILE, name)