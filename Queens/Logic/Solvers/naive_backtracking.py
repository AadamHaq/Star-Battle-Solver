"""
Inspiration for the code:
- N-Queens Solution (famous backtracking problem) Leetcode 51
    - Used https://www.youtube.com/watch?v=Ph95IHmRp5M solution from Neetcode
    - https://www.geeksforgeeks.org/python-program-for-n-queen-problem-backtracking-3/
- https://informatika.stei.itb.ac.id/~rinaldi.munir/Stmik/2019-2020/Makalah/stima2020k2-035.pdf
"""

import time
from typing import List, Set, Tuple


def is_valid(
    row: int,
    col: int,
    board: List[List[int]],
    queens: Set[Tuple[int, int]],  # (row,col) pair
    used_cols: Set[int],
    used_colours: Set[int],
) -> bool:
    """
    Function: is_valid sees if a Queen can be placed

    Args:
        row: int between [0, N-1]
        col: int between [0, N-1]
        board: List of Lists of Integers between [0, N-1]
        queens: A set of tuples
        used_cols: A set of integers that are between [0, N-1]
        used_colours: A set of integers that are between [0, N-1]

    Description: Checks to see if it is valid to make a move. It checks:
        - If a column already has a Queen
        - If a coloured colour already has a Queen
        - If there are any Queens in it's 3x3 area

    N.B: We don't check row as rows are done sequentially and when a Queen is placed, we move to the next row

    Returns: Boolean -> True if can place a queen
    """

    # Column Check
    if col in used_cols:
        return False

    # Coloured colour Check
    colour = board[row][col]
    if colour in used_colours:
        return False

    # Check 3x3 neighborhood (adjacent cells)
    for adj_row in (-1, 0, 1):  # Next row used to look at top right cell
        for adj_col in (-1, 0, 1):
            if adj_row == 0 and adj_col == 0:
                continue  # Skip looking at current position
            neighbour_row, neighbour_col = row + adj_row, col + adj_col
            if (neighbour_row, neighbour_col) in queens:
                return False  # False if there is a neighbouring queen

    return True  # Passed all checks


def backtracking(board: List[List[int]], N: int = 0) -> List[Tuple[int, int]]:
    """
    Function: backtracking solution

    Args:
        board: List of Lists of Integers between [0, N-1]
        N: int for length of board. Default is 0

    Description: Solves the board using backtracking (recursive algorithm)

    Returns: List[Tuple(int, int)]] -> rows and columns of queen solutions
    """
    queens: Set[Tuple[int, int]] = set()
    used_cols: Set[int] = set()
    used_colours: Set[int] = set()

    def _backtrack(row: int) -> List[Tuple[int, int]]:
        if row == N:
            return list(queens)

        for col in range(N):
            if is_valid(row, col, board, queens, used_cols, used_colours):
                queens.add((row, col))
                used_cols.add(col)
                used_colours.add(board[row][col])

                result = _backtrack(row + 1)
                if result:
                    return result

                queens.remove((row, col))
                used_cols.remove(col)
                used_colours.remove(board[row][col])

        return []

    return _backtrack(0)


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
    backtracking(test_board, N=8)
    end = time.time()
    print(f"Backtracking Time: {end - start:4f} seconds \n")

    # Instant - 0.00000s
    # Better than Linear Programming currently
