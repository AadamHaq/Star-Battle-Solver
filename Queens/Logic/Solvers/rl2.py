"""
Q-Learning for LinkedIn Queens (action = place a queen in ANY empty cell).
Behavior:
 - Action space = all (row, col) cells flattened to integer actions.
 - If the agent places an invalid queen (violates adjacency / row/col / colour) -> reward = -1, episode restarts.
 - If the agent places a legal queen but that (row,col) is NOT in the solved_board -> reward = -1, episode restarts.
 - If the agent places a legal queen that IS in the solved_board -> reward = +1 and episode continues.
 - If the agent places all queens in the solved_board (successful episode) -> extra +10 reward.
 - Training iterates through all boards in metadata folder and saves Q-table to RL_Resources/Weights/queens_q_table.pkl
"""

import numpy as np
import os
import glob
import ast
import pickle
from typing import List, Tuple, Dict

class QueensQLearning:
    def __init__(self, epsilon: float = 0.2, alpha: float = 0.1, gamma: float = 0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict[str, Dict[int, float]] = {}

    def state_key(self, board: List[List[int]], queens: List[Tuple[int,int]]):
        # Represent board as a flattened tuple and queens as a sorted tuple for stable keys
        flat_board = tuple(x for row in board for x in row)
        # sorted queens ensures order-invariance
        queens_key = tuple(sorted(queens))
        return (len(board), flat_board, queens_key)  # include size to avoid collisions

    def all_actions(self, board: List[List[int]]) -> List[int]:
        N = len(board)
        return [r * N + c for r in range(N) for c in range(N)]

    def action_to_rc(self, action: int, N: int):
        return divmod(action, N)

    def is_move_valid(self, board: List[List[int]], queens: List[Tuple[int,int]], r: int, c: int) -> bool:
        N = len(board)
        # can't place where a queen already is
        if (r, c) in queens:
            return False
        # row/column conflict
        for qr, qc in queens:
            if qr == r or qc == c:
                return False
            # adjacency including diagonals
            if abs(qr - r) <= 1 and abs(qc - c) <= 1:
                return False
            # diagonal beyond adjacency? original problem forbids same row/col and adjacency; we keep those rules.
        # color constraint
        placed_colours = {board[qr][qc] for qr, qc in queens}
        if board[r][c] in placed_colours:
            return False
        return True

    def get_valid_actions(self, board: List[List[int]], queens: List[Tuple[int,int]]) -> List[int]:
        """Return all empty cells (we don't filter legality here because the agent is permitted to try any cell;
           training penalizes illegal choices)."""
        N = len(board)
        occupied = set(queens)
        return [a for a in self.all_actions(board) if self.action_to_rc(a, N) not in occupied]

    def get_action(self, state_key, valid_actions: List[int]) -> int:
        if not valid_actions:
            return -1
        if np.random.random() < self.epsilon or state_key not in self.q_table:
            return int(np.random.choice(valid_actions))
        # choose best q among valid actions
        qdict = self.q_table.get(state_key, {})
        q_values = [qdict.get(a, 0.0) for a in valid_actions]
        max_q = max(q_values)
        candidates = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return int(np.random.choice(candidates))

    def update_q(self, state_key, action: int, reward: float, next_state_key, next_valid_actions: List[int]):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        current_q = self.q_table[state_key].get(action, 0.0)

        next_max_q = 0.0
        if next_valid_actions and next_state_key in self.q_table:
            next_qs = [self.q_table[next_state_key].get(a, 0.0) for a in next_valid_actions]
            next_max_q = max(next_qs) if next_qs else 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

    def train_on_board(self, board: List[List[int]], solution: List[Tuple[int,int]], episodes: int = 100):
        N = len(board)
        solution_set = set(solution)

        for ep in range(episodes):
            queens = []
            state = self.state_key(board, queens)

            # allow up to N placements per successful episode but episodes will restart on a 'wrong' placement
            for step in range(N):
                valid_actions = self.get_valid_actions(board, queens)
                if not valid_actions:
                    break

                action = self.get_action(state, valid_actions)
                if action == -1:
                    break

                r, c = self.action_to_rc(action, N)

                # Check legality wrt constraints
                legal = self.is_move_valid(board, queens, r, c)

                if not legal:
                    # Illegal placement -> strong negative reward and episode ends (restart next episode)
                    reward = -1.0
                    next_state = self.state_key(board, queens)  # no change because move rejected
                    self.update_q(state, action, reward, next_state, [])
                    break  # end episode early on illegal move

                # If placement is legal but not part of known solution -> negative reward and restart (per instruction)
                if (r, c) not in solution_set:
                    reward = -1.0
                    next_state = self.state_key(board, queens)  # still before adding the wrong queen
                    self.update_q(state, action, reward, next_state, [])
                    break

                # placement legal AND in solution -> positive reward and continue
                queens.append((r, c))
                reward = 1.0

                # If we've placed the full solution (all solution coords), give big reward and finish episode
                if set(queens) == solution_set:
                    # success
                    reward += 10.0  # bonus for completing full solution
                    next_state = self.state_key(board, queens)
                    self.update_q(state, action, reward, next_state, [])
                    break

                # otherwise, continue to next step
                next_state = self.state_key(board, queens)
                next_valid = self.get_valid_actions(board, queens)
                self.update_q(state, action, reward, next_state, next_valid)

                # prepare for next loop iteration
                state = next_state

    def solve_board(self, board: List[List[int]], max_attempts: int = 1000, greedy: bool = True) -> List[Tuple[int,int]]:
        """Attempt to solve a board using learned Q-values. We'll try multiple episodes and return first successful board found."""
        N = len(board)
        for attempt in range(max_attempts):
            queens = []
            state = self.state_key(board, queens)
            for step in range(N):
                valid_actions = self.get_valid_actions(board, queens)
                if not valid_actions:
                    break
                # pick greedy action (no exploration) if asked, else with epsilon
                if greedy and state in self.q_table:
                    qdict = self.q_table[state]
                    q_values = [qdict.get(a, 0.0) for a in valid_actions]
                    max_q = max(q_values)
                    candidates = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                    action = int(np.random.choice(candidates))
                else:
                    action = self.get_action(state, valid_actions)

                r, c = self.action_to_rc(action, N)
                if not self.is_move_valid(board, queens, r, c):
                    break  # attempt failed, try a new attempt
                queens.append((r, c))
                state = self.state_key(board, queens)

            if len(queens) == N:
                return queens
        return []  # failed to find a solution

    def save_weights(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)


def load_training_data(metadata_path: str):
    training_data = []
    txt_files = glob.glob(os.path.join(metadata_path, "*.txt"))
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            content = f.read()
        board_str = content[content.find("queens_board = "):content.find("solved_board")].strip()
        solution_str = content[content.find("solved_board = "):].strip()
        board = ast.literal_eval(board_str.split(" = ", 1)[1])
        solution = ast.literal_eval(solution_str.split(" = ", 1)[1])
        training_data.append((board, solution))
    return training_data


if __name__ == "__main__":
    agent = QueensQLearning(epsilon=0.2, alpha=0.1, gamma=0.9)

    metadata_path = os.path.join(os.path.dirname(__file__), "RL_Resources", "Training_Data", "Metadata")
    training_data = load_training_data(metadata_path)

    # train on every board (you said there are 300+ boards)
    episodes_per_board = 300  # increase/decrease depending on compute/time
    for board, solution in training_data:
        print(f"Training on {len(board)}x{len(board)} board...")
        agent.train_on_board(board, solution, episodes=episodes_per_board)

    weights_path = os.path.join(os.path.dirname(__file__), "RL_Resources", "Weights", "queens_q_table.pkl")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    agent.save_weights(weights_path)

    # quick test: try solving first board
    if training_data:
        test_board = training_data[0][0]
        found = agent.solve_board(test_board, max_attempts=500, greedy=True)
        print("Found solution:", found)
