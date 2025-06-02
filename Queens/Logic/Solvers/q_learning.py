"""
Q-Learning implementation for the Queens puzzle solver.
This implementation learns from training data to solve queen placement puzzles
of various sizes while respecting color region constraints, row and column constraints
and no queen can be adjacent to another.
"""

import os
import ast
import numpy as np
import random

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 10000
invalid_move_penalty = -10
valid_move_reward = 1
success_reward = 100

# File paths
metadata_dir = "Queens\Logic\Solvers\RL_Resources\Training_Data\Metadata"
weights_dir = "Queens\Logic\Solvers\RL_Resources/Weights"
os.makedirs(weights_dir, exist_ok=True)

# Parse board from uploaded file
def parse_board_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    board_str = content.split("solved_board")[0].strip()
    board_data = ast.literal_eval(board_str.split(" = ")[1])
    return np.array(board_data)

# Get state key for Q-table
def get_q_table_key(board):
    return ''.join(map(str, board.flatten()))

# Check if placing a queen is valid
def is_valid(board, r, c):
    N = board.shape[0]
    if board[r, c] == 1:
        return False
    for i in range(N):
        if board[i, c] == 1 or board[r, i] == 1:
            return False
    for i in range(-N, N):
        if 0 <= r+i < N and 0 <= c+i < N and board[r+i, c+i] == 1:
            return False
        if 0 <= r+i < N and 0 <= c-i < N and board[r+i, c-i] == 1:
            return False
    return True

# Load boards
file_list = ["queens3.txt", "queens8.txt", "queens11.txt"]
training_boards = [parse_board_from_file(os.path.join(metadata_dir, f)) for f in file_list]

# Q-learning loop
Q = {}
for ep in range(episodes):
    board = random.choice(training_boards).copy()
    N = board.shape[0]
    state_key = get_q_table_key(board)
    done = False
    steps = 0

    while not done and steps < N * N:
        if state_key not in Q:
            Q[state_key] = np.zeros((N, N))

        if random.random() < epsilon:
            action = (random.randint(0, N-1), random.randint(0, N-1))
        else:
            action = np.unravel_index(np.argmax(Q[state_key]), (N, N))

        r, c = action
        steps += 1

        if is_valid(board, r, c):
            board[r, c] = 1
            reward = valid_move_reward
            next_state_key = get_q_table_key(board)
            if np.sum(board) == N:
                reward = success_reward
                done = True
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros((N, N))
            Q[state_key][r, c] += alpha * (reward + gamma * np.max(Q[next_state_key]) - Q[state_key][r, c])
            state_key = next_state_key
        else:
            reward = invalid_move_penalty
            Q[state_key][r, c] += alpha * (reward - Q[state_key][r, c])
            done = True

# Save the Q-table
np.save(os.path.join(weights_dir, "queens_q_table.npy"), Q)

print("Saved")

# Load the Q-table if needed
Q = np.load(os.path.join(weights_dir, "queens_q_table.npy"), allow_pickle=True).item()

# Test board (let's use queens8.txt as an example)
test_board = parse_board_from_file(os.path.join(metadata_dir, "queens8.txt"))
N = test_board.shape[0]
board = np.zeros((N, N), dtype=int)
state_key = get_q_table_key(board)
solution = []

for _ in range(N * N):
    if state_key not in Q:
        print("State not found in Q-table. Aborting.")
        break

    action = np.unravel_index(np.argmax(Q[state_key]), (N, N))
    r, c = action

    if not is_valid(board, r, c):
        print(f"Invalid move at ({r}, {c}). Aborting.")
        break

    board[r, c] = 1
    solution.append((r, c))
    if len(solution) == N:
        print("Found solution:")
        print(solution)
        break

    state_key = get_q_table_key(board)
else:
    print("Failed to find complete solution.")