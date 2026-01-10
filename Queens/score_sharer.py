from Logic.game_scraper import initialise_driver
from Logic.plot_scores import plot_graph, write_csv
from Logic.scrape_scores_from_messages import score_scraper
from Logic.upload_and_share_plot import upload_plot


def main(cookie_file, name):
    driver = initialise_driver(cookie_file)
    results = score_scraper(driver, name)
    print(results)
    write_csv(results)
    plot_graph()
    upload_plot(driver)

    driver.quit()


if __name__ == "__main__":
    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Queens + Zip Daily"
    main(COOKIE_FILE, name)
