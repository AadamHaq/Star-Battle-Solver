"""
Inspiration for code
- Z3 solution: https://medium.com/towards-data-engineering/solving-linkedins-queens-puzzle-with-code-6f1c3aa23a40
- https://medium.com/data-science/using-linear-equations-llm-to-solve-linkedin-queens-game-fe9802d997e9

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