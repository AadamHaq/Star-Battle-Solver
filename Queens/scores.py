from Logic.scraper import initialise_driver
from Logic.scrape_scores import score_scraper
from Logic.plotter import write_csv, plot_graph
from Logic.upload_plot import upload_plot

def main(cookie_file, name):
    driver = initialise_driver(cookie_file)
    results = score_scraper(driver, name)
    write_csv(results)
    plot_graph()
    upload_plot(driver)

    driver.quit()



if __name__ == "__main__":
    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Queens + Zip Daily"
    main(COOKIE_FILE, name)