"""
Inspiration for code
- Z3 solution: https://medium.com/towards-data-engineering/solving-linkedins-queens-puzzle-with-code-6f1c3aa23a40
- PuLP solution: https://medium.com/data-science/using-linear-equations-llm-to-solve-linkedin-queens-game-fe9802d997e9
Aim: Try our own Z3 and PuLP solutions and see what works better from my code

Methodology:

Why was linear programming an option?
In Queens, the rules are simple. We need one Queen in every row, column and region, with no Queens adjacent to each other. This can be represented as the following:
    - One Queen in every row:
        x_11 + x_12 + ... + x_1n = 1 (Repeat logic for n rows)
    - One Queen in every column:
        x_11 + x_21 + ... + x_n1 = 1  (Repeat logic for n columns)

Example board:
[0, 1, 1, 2, 3, 3, 3, 4]
[0, 0, 1, 2, 2, 3, 3, 4]
[5, 0, 0, 0, 0, 3, 3, 4]
[5, 5, 0, 0, 0, 0, 3, 4]
[6, 6, 0, 0, 0, 0, 3, 0]
[7, 6, 0, 0, 0, 0, 0, 0]
[7, 7, 7, 0, 0, 0, 0, 0]
[7, 7, 7, 7, 7, 0, 0, 0]

    - One Queen in every region (region colour 1 used as an example):
        x_12 + x_13 + x_31 = 1
    - No Queens adjacent to each other:
        For position x_pq we have:
            x_pq + x_{p+1}q <= 1
            x_pq + x_{p-1}q <= 1
            x_pq + x_p{q+1} <= 1
            x_pq + x_p{q-1} <= 1
            x_pq + x_{p+1}{q+1} <= 1
            x_pq + x_{p-1}{q+1} <= 1
            x_pq + x_{p+1}{q-1} <= 1
            x_pq + x_{p-1}{q-1} <= 1
"""
from z3 import Bool, Solver, Sum, If, Or, Not, sat
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus
import time


def solve_queens_puzzle_z3(board):
    n = len(board)
    solver = Solver()

    # Create Boolean variables: Q[i][j] is True if there's a queen at cell (i, j)
    Q = [[Bool(f"Q_{i}_{j}") for j in range(n)] for i in range(n)]

    # 1. One queen per row
    for i in range(n):
        solver.add(Sum([If(Q[i][j], 1, 0) for j in range(n)]) == 1) # Sum of row == 1

    # 2. One queen per column
    for j in range(n):
        solver.add(Sum([If(Q[i][j], 1, 0) for i in range(n)]) == 1) # Sum of col == 1

    # 3. One queen per region
    region_ids = set(i for i in range(n))
    for region in region_ids:
        region_cells = [(i, j) for i in range(n) for j in range(n) if board[i][j] == region] # All cells in one region
        solver.add(Sum([If(Q[i][j], 1, 0) for (i, j) in region_cells]) == 1) # Must sum to 1

    # 4. No adjacent queens (8-neighbors)
    for i in range(n):
        for j in range(n):
            neighbors = [
                (i + dx, j + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if not (dx == 0 and dy == 0) # Not itself
            ]
            for ni, nj in neighbors:
                if 0 <= ni < n and 0 <= nj < n:
                    solver.add(Or(Not(Q[i][j]), Not(Q[ni][nj]))) # Only up to one can be a queen

    # Solve
    if solver.check() == sat: # If there is a solution (There always should be)
        model = solver.model()
        solution = [[1 if model.evaluate(Q[i][j]) else 0 for j in range(n)] for i in range(n)] # 1 where Queen is true
        print("Solution found:")
        for row in solution:
            print(" ".join(str(cell) for cell in row))
        return solution
    else:
        print("No solution exists.")
        return None


def solve_queens_puzzle_pulp(board):
    n = len(board)
    model = LpProblem("Queens", LpMinimize) # Minimise Integer Linear Programming Problem

    # Variables: x[i][j] = 1 if there's a queen at (i, j), 0 otherwise
    x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(n)] for i in range(n)]

    # Objective: No real optimisation goal, so we minimize 0
    model += 0

    # 1. One queen per row
    for i in range(n):
        model += lpSum(x[i][j] for j in range(n)) == 1 # Similar to Z3

    # 2. One queen per column
    for j in range(n):
        model += lpSum(x[i][j] for i in range(n)) == 1 # Similar to Z3

    # 3. One queen per region
    region_ids = set(i for i in range(n))
    for region in region_ids:
        model += lpSum(x[i][j] for i in range(n) for j in range(n) if board[i][j] == region) == 1 # Similar to Z3

    # 4. No adjacent queens
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    
    for i in range(n):
        for j in range(n):
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < n:
                    model += x[i][j] + x[ni][nj] <= 1 # Cell + Neighbour has at most 1 Queen

    # Solve
    status = model.solve()
    if LpStatus[status] == "Optimal":
        solution = [[int(x[i][j].varValue) for j in range(n)] for i in range(n)] # varValue so it's an int (not as 1.0)
        print("Solution found:")
        for row in solution:
            print(" ".join(str(cell) for cell in row))
        return solution
    else:
        print("No solution found.")
        return None

if __name__ == "__main__":
    test_board = [
    [0, 1, 1, 2, 3, 3, 3, 4],
    [0, 0, 1, 2, 2, 3, 3, 4],
    [5, 0, 0, 0, 0, 3, 3, 4],
    [5, 5, 0, 0, 0, 0, 3, 4],
    [6, 6, 0, 0, 0, 0, 3, 0],
    [7, 6, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 0, 0, 0],
    ]
    start = time.time()
    solve_queens_puzzle_z3(test_board)
    end = time.time()
    print(f"Z3 Time: {end - start:.4f} seconds \n")

    start = time.time()
    solve_queens_puzzle_pulp(test_board)
    end = time.time()
    print(f"PuLP Time: {end - start:.4f} seconds \n")

    # Verdict: Z3 was marginally faster so we will go with that


