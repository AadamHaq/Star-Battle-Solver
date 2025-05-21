import os
from naive_backtracking import backtracking  # Assumes naive_backtracking.py is in the same folder or PYTHONPATH

directory = r"Queens\Logic\Solvers\RL_Resources\Training_Data\Metadata"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)

        with open(filepath, 'r') as file:
            content = file.read()

        # Parse board from file (assumes variable is named 'queens_board')
        try:
            local_vars = {}
            exec(content, {}, local_vars)
            queens_board = local_vars.get("queens_board")

            if not queens_board:
                print(f"Could not find 'queens_board' in {filename}")
                continue

            N = len(queens_board)
            solution = backtracking(queens_board, N)

            # Append the solution to the file
            with open(filepath, 'a') as file:
                file.write(f"\nsolved_board = {solution}\n")
            print(f"Solved and updated {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
