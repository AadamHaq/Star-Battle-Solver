#!/usr/bin/env python3
"""
rl3_improved.py

Usage:
    python Queens\Logic\Solvers\rl3.py --mode tabular
    python Queens\Logic\Solvers\rl3.py --mode dqn

Put your training .txt files in:
  RL_Resources/Training_Data/Metadata (next to this script)

The script will save weights to:
  RL_Resources/Weights/...
"""
import os
import glob
import ast
import pickle
import random
from typing import List, Tuple, Dict, Set
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "RL_Resources", "Training_Data", "Metadata")
WEIGHTS_DIR = os.path.join(BASE_DIR, "RL_Resources", "Weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# -------------------------
# Training data loader
# -------------------------
def load_training_data(metadata_path: str = METADATA_DIR):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata folder not found at: {metadata_path}")
    txt_files = sorted(glob.glob(os.path.join(metadata_path, "*.txt")))
    training = []
    max_color = 0
    max_N = 0
    for p in txt_files:
        with open(p, "r") as f:
            content = f.read()
        try:
            start_idx = content.find("queens_board =")
            mid_idx = content.find("solved_board =")
            board = ast.literal_eval(content[start_idx:mid_idx].split("=",1)[1].strip())
            solved = ast.literal_eval(content[mid_idx:].split("=",1)[1].strip())
            training.append((board, solved))
            max_N = max(max_N, len(board))
            # compute max color id seen
            for row in board:
                for c in row:
                    if isinstance(c, int):
                        max_color = max(max_color, c)
        except Exception as e:
            print("Warning: failed to parse", p, e)
    # curriculum: small -> large
    training.sort(key=lambda x: len(x[0]))
    return training, max_N, max_color

# -------------------------
# Utilities
# -------------------------
def action_to_rc(action:int, N:int)->Tuple[int,int]:
    return divmod(int(action), N)

def flatten_board(board:List[List[int]]):
    return [c for r in board for c in r]

# -------------------------
# Tabular Q-learning with remove-last action
# -------------------------
class TabularAgent:
    def __init__(self,
                 initial_epsilon=0.25,
                 min_epsilon=0.01,
                 epsilon_decay=0.9999,
                 alpha=0.1,
                 gamma=0.95,
                 max_illegal=3,
                 reward_valid=8.0,
                 reward_invalid=-2.0,
                 reward_complete=500.0,
                 step_penalty=-0.005,
                 demo_seed=5):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict = {}
        self.max_illegal = max_illegal
        self.reward_valid = reward_valid
        self.reward_invalid = reward_invalid
        self.reward_complete = reward_complete
        self.step_penalty = step_penalty
        self.demo_seed = demo_seed  # number of imitation episodes to seed

    def state_key(self, N:int, queens:List[Tuple[int,int]], used_colours:Set[int]):
        # compact transferable key
        return (N, tuple(sorted(queens)), tuple(sorted(used_colours)))

    def all_actions(self, N:int):
        # 0..N*N-1 => place at r,c ; N*N => remove last placed queen
        return list(range(N*N + 1))

    def is_legal(self, board:List[List[int]], queens:List[Tuple[int,int]], r:int, c:int)->bool:
        N = len(board)
        if (r,c) in queens:
            return False
        for qr,qc in queens:
            if qr == r or qc == c:
                return False
            if abs(qr - r) <= 1 and abs(qc - c) <= 1:
                return False
        used = {board[qr][qc] for qr,qc in queens} if queens else set()
        if board[r][c] in used:
            return False
        return True

    def get_unoccupied_actions(self, board, queens):
        N = len(board)
        occupied = set(queens)
        place_actions = [r*N+c for r in range(N) for c in range(N) if (r,c) not in occupied]
        return place_actions + [N*N]  # include remove-last

    def choose_action(self, state_key, valid_actions):
        if (not valid_actions) or (np.random.random() < self.epsilon) or (state_key not in self.q_table):
            return int(random.choice(valid_actions)) if valid_actions else -1
        qdict = self.q_table.get(state_key, {})
        q_vals = [qdict.get(a, 0.0) for a in valid_actions]
        max_q = max(q_vals)
        candidates = [a for a,q in zip(valid_actions,q_vals) if q==max_q]
        return int(random.choice(candidates))

    def update_q(self, state_key, action, reward, next_state_key, next_valid_actions):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        cur_q = self.q_table[state_key].get(action, 0.0)
        next_max = 0.0
        if next_valid_actions and next_state_key in self.q_table:
            next_qs = [self.q_table[next_state_key].get(a,0.0) for a in next_valid_actions]
            next_max = max(next_qs) if next_qs else 0.0
        new_q = cur_q + self.alpha * (reward + self.gamma * next_max - cur_q)
        self.q_table[state_key][action] = new_q

    # Demo-seeding using solved_board trajectory (small number of episodes)
    def seed_with_demonstration(self, board, solved_board, n_demos=5):
        N = len(board)
        if not solved_board:
            return
        for _ in range(n_demos):
            queens=[]
            for (r,c) in solved_board:
                state = self.state_key(N, queens, {board[q[0]][q[1]] for q in queens} if queens else set())
                action = r*N + c
                # reward artificially high to seed q-values
                reward = self.reward_valid * 2
                queens.append((r,c))
                next_state = self.state_key(N, queens, {board[q[0]][q[1]] for q in queens})
                next_valid = self.get_unoccupied_actions(board, queens)
                self.update_q(state, action, reward, next_state, next_valid)
            # final step bonus
            final_state = self.state_key(N, queens, {board[q[0]][q[1]] for q in queens})
            # dummy update for terminal
            self.update_q(final_state, N*N, self.reward_complete*2, final_state, [])

    def train_on_board(self, board, solved_board, episodes_cap=200000, mastery_required=1, verbose=True):
        N = len(board)
        episodes = 0
        solves_in_a_row = 0
        best_progress = 0
        episodes_since_improvement = 0

        # reset exploration for new board
        self.epsilon = self.initial_epsilon

        # seed with demos (light supervision) — comment out to be purely unsupervised
        if self.demo_seed and solved_board:
            self.seed_with_demonstration(board, solved_board, n_demos=self.demo_seed)

        while episodes < episodes_cap and solves_in_a_row < mastery_required:
            episodes += 1
            queens=[]
            state = self.state_key(N, queens, set())
            illegal_count = 0
            episode_best = 0

            for step in range(3*N):  # allow up to 3*N steps (placements+removals)
                valid_actions = self.get_unoccupied_actions(board, queens)
                if not valid_actions:
                    break
                action = self.choose_action(state, valid_actions)
                if action == -1:
                    break

                # remove-last action
                if action == N*N:
                    if not queens:
                        reward = self.reward_invalid + self.step_penalty
                        next_state = state
                        self.update_q(state, action, reward, next_state, [])
                        illegal_count += 1
                        if illegal_count > self.max_illegal:
                            break
                        continue
                    # remove last queen (repair)
                    removed = queens.pop()
                    next_state = self.state_key(N, queens, {board[q[0]][q[1]] for q in queens} if queens else set())
                    reward = -0.5 + self.step_penalty  # small penalty for removal
                    next_valid = self.get_unoccupied_actions(board, queens)
                    self.update_q(state, action, reward, next_state, next_valid)
                    state = next_state
                    continue

                r,c = divmod(action, N)
                legal = self.is_legal(board, queens, r, c)
                step_penalty = self.step_penalty

                if not legal:
                    illegal_count += 1
                    reward = self.reward_invalid + step_penalty
                    next_state = self.state_key(N, queens, {board[q[0]][q[1]] for q in queens} if queens else set())
                    self.update_q(state, action, reward, next_state, [])
                    if illegal_count > self.max_illegal:
                        break
                    else:
                        continue

                # legal placement
                queens.append((r,c))
                progress = len(queens)
                episode_best = max(episode_best, progress)
                used_cols = {board[q[0]][q[1]] for q in queens}
                next_state = self.state_key(N, queens, used_cols)
                next_valid = self.get_unoccupied_actions(board, queens)
                partial = (progress / N)
                reward = self.reward_valid + partial * 1.0 + step_penalty

                if progress == N:
                    reward += self.reward_complete
                    self.update_q(state, action, reward, next_state, [])
                    solves_in_a_row += 1
                    break

                self.update_q(state, action, reward, next_state, next_valid)
                state = next_state

            # track improvement
            if episode_best > best_progress:
                best_progress = episode_best
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1

            # if stuck, temporarily boost exploration
            if episodes_since_improvement > 20000:
                old_eps = self.epsilon
                self.epsilon = max(self.epsilon, 0.5)
                if verbose:
                    print(f"[board {N}x{N}] stuck -> boosting epsilon to {self.epsilon} (was {old_eps})")
                episodes_since_improvement = 0  # give it some time to explore

            # if solved reset flag
            if solves_in_a_row == 0 or best_progress < N:
                solves_in_a_row = 0

            # decay
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if verbose and episodes % 1000 == 0:
                print(f"[board {N}x{N}] episodes={episodes:,} eps={self.epsilon:.4f} best_progress={best_progress}/{N}")

        success = solves_in_a_row >= mastery_required
        return success, best_progress

    def save(self, path=None):
        path = path or os.path.join(WEIGHTS_DIR, "q_table_tabular.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print("Saved tabular q_table:", path)

    def load(self, path=None):
        path = path or os.path.join(WEIGHTS_DIR, "q_table_tabular.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            print("Loaded tabular q_table:", path)

# -------------------------
# DQN Agent (PyTorch)
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = map(np.array, zip(*batch))
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buffer)

# -------------------------
# Training orchestrators
# -------------------------
def run_tabular():
    training, _, _ = load_training_data(METADATA_DIR)
    agent = TabularAgent()
    agent.load = lambda path=None: None  # placeholder in case old load exists
    # attempt each board
    total = len(training)
    solved_total=0
    for i,(board, solved_board) in enumerate(training, start=1):
        print(f"\n--- Tabular: board {i}/{total} ({len(board)}x{len(board)}) ---")
        success, best = agent.train_on_board(board, solved_board, episodes_cap=200000, mastery_required=1, verbose=True)
        if success:
            solved_total += 1
            print("Solved.")
        else:
            print("Not solved. Best progress:", best, "/", len(board))
        agent.save(os.path.join(WEIGHTS_DIR, f"q_table_tabular_{i}.pkl"))
    print(f"\nTabular training done. solved {solved_total}/{total}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    run_tabular()
