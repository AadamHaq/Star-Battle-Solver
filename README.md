# LinkedIn Queens Solver

The aim of the project is to create an autosolver for the starbattle game. This will be done in two stages:

- LinkedIn Queens Solver (1 star - Star Battle)
- Star Battle solver (2 and 3 stars)

Star Battle is a generalisation of the LinkedIn Queens game. An example can be found [here](https://www.puzzle-star-battle.com/)

---
## LinkedIn Queens Solver

To solve LinkedIn Queens, I want to use several techniques:
- Backtracking (Naive solution as a baseline)
- Linear Programming
  - This will constitute breaking the game into its pure logic and creating linear equations
- Reinforcement Learning Approach
  - Will train daily and then test using saved weights

Additionally, I will scrape the current day's puzzle and solve it, before then seeing if it can be inputted to beat some of my friend's times (anonymised to spare any embarrassment!)

### How to run

- For first time use run `Queens\Logic\get_cookies.py` to got the cookies
- Run `Queens\queens_solver.ipynb` to create a solution
  - Input what method you would like to solve: Backtracking, Linear Programming, Reinforcement Learning
  - Individual Files ran in order are:
    - `Queens\Logic\scraper.py`
      - Currently scrapes using selenium, but I'll experiment with computer vision too
    - Solver choice used (all can be found in `Queens\Logic\Solvers`)
    - `Queens\Logic\inputter.py`

### Learnings

This project was shorter than initially intended! The backtracking algorithm performed much better than initially thought, so left little reason to complete the other methods in practice; however these will be done anyway for fun.

Key learnings:

- Using a scraper and inputter with selenium
- Improve backtracking algorithm skills
- Problem Solving to find ways to reduce time complexity

Next Steps:

- ~~Complete Linear Programming Solution~~
- ~~Create Computer Vision Scraper (never used before)~~
- Create RL solution after training data obtained
- ~~Add a way to share my time onto a group chat with my friends~~
- ~~Scrape their times daily and add them to a csv rather than doing this manually~~
- ~~Plotter to create a graph of everyone's times per day~~
- ~~Implement CI/CD Pipelines and pre-commit~~
- ~~Convert project to uv~~