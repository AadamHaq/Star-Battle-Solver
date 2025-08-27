#!/usr/bin/env python3
"""
rl3_dqn_unsupervised.py

Unsupervised DQN solver for the LinkedIn Queens variant (curriculum learning).
- Purely unsupervised: DOES NOT use solved_board for training.
- Curriculum: trains boards sorted by size (small -> large).
- Action space: place at any cell (0..N*N-1) + remove-last action (index N*N).
- State encoding: two channels padded to max_N across dataset:
    1) color id (normalized)
    2) queen mask (0/1)
- Experience replay + target network + epsilon-greedy.
- Save / load model from RL_Resources/Weights/

Usage:
    python rl3_dqn_unsupervised.py \
        --episodes-per-board 3000 \
        --batch-size 128 \
        --device auto \
        --boards-limit 50

Defaults are conservative; tuning may be necessary.
"""

import os
import glob
import ast
import argparse
import random
import math
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "RL_Resources", "Training_Data", "Metadata")
WEIGHTS_DIR = os.path.join(BASE_DIR, "RL_Resources", "Weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Environment / reward shaping (tune these)
REWARD_VALID = 6.0
REWARD_INVALID = -1.0
REWARD_COMPLETE = 400.0
STEP_PENALTY = -0.005

# Action: place (0..N*N-1), remove_last = N*N

# DQN defaults (tuneable via CLI)
DEFAULT_EPISODES_PER_BOARD = 3000
DEFAULT_MAX_STEPS_MULT = 3         # max steps per episode = max_steps_mult * N
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_GAMMA = 0.99
DEFAULT_BUFFER_CAP = 200_000
DEFAULT_TARGET_UPDATE = 1000
INITIAL_EPS = 0.25
MIN_EPS = 0.01
EPS_DECAY = 0.99995

DEVICE_CHOICES = ("cpu", "cuda", "auto")

# -------------------- Data loader --------------------
def load_training_data(metadata_path: str = METADATA_DIR):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata folder not found at: {metadata_path}")
    files = sorted(glob.glob(os.path.join(metadata_path, "*.txt")))
    training = []
    max_N = 0
    max_color = 0
    for p in files:
        with open(p, "r") as f:
            content = f.read()
        try:
            start_idx = content.find("queens_board =")
            mid_idx = content.find("solved_board =")
            if start_idx == -1 or mid_idx == -1:
                raise ValueError("file missing queens_board/solved_board markers")
            board = ast.literal_eval(content[start_idx:mid_idx].split("=", 1)[1].strip())
            solved = ast.literal_eval(content[mid_idx:].split("=", 1)[1].strip())
            training.append((board, solved))
            max_N = max(max_N, len(board))
            for row in board:
                for c in row:
                    if isinstance(c, int):
                        max_color = max(max_color, c)
        except Exception as e:
            print("Warning: failed to parse", p, e)
    # curriculum: smallest boards first
    training.sort(key=lambda x: len(x[0]))
    return training, max_N, max_color

# -------------------- Utilities --------------------
def is_legal_placement(board: List[List[int]], queens: List[Tuple[int,int]], r: int, c: int) -> bool:
    N = len(board)
    if (r, c) in queens:
        return False
    for qr, qc in queens:
        if qr == r or qc == c:
            return False
        if abs(qr - r) <= 1 and abs(qc - c) <= 1:
            return False
    placed_colours = {board[qr][qc] for qr, qc in queens} if queens else set()
    if board[r][c] in placed_colours:
        return False
    return True

def pad_and_encode(board: List[List[int]], queens: List[Tuple[int,int]], max_N: int, max_color: int):
    """
    Returns a float32 vector: [color_channel_flat, queen_mask_flat]
    color_channel: padded with -1 for off-grid cells, normalized by (max_color or 1)
    queen_mask: 1 if queen placed, else 0 (padded 0 for off-grid)
    """
    N = len(board)
    color = np.full((max_N, max_N), -1.0, dtype=np.float32)
    mask = np.zeros((max_N, max_N), dtype=np.float32)
    denom = max(1, max_color)
    for r in range(N):
        for c in range(N):
            color[r, c] = float(board[r][c]) / denom
    for (r, c) in queens:
        mask[r, c] = 1.0
    return np.concatenate([color.ravel(), mask.ravel()]).astype(np.float32)

# -------------------- Replay buffer --------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

# -------------------- Network --------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- DQN Agent --------------------
class DQNAgent:
    def __init__(self, max_N, max_color, device: str = "cpu",
                 lr=DEFAULT_LR, gamma=DEFAULT_GAMMA, batch_size=DEFAULT_BATCH_SIZE,
                 buffer_cap=DEFAULT_BUFFER_CAP, target_update=DEFAULT_TARGET_UPDATE):
        self.max_N = max_N
        self.max_color = max_color
        self.input_dim = max_N * max_N * 2
        self.action_dim = max_N * max_N + 1  # place actions + remove-last
        self.device = torch.device(device)
        self.net = MLP(self.input_dim, self.action_dim).to(self.device)
        self.target = MLP(self.input_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_cap)
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0
        self.eps = INITIAL_EPS
        self.min_eps = MIN_EPS
        self.eps_decay = EPS_DECAY

    def choose_action(self, state_input: np.ndarray, valid_actions: List[int]):
        if (np.random.random() < self.eps) and len(valid_actions) > 0:
            return int(random.choice(valid_actions))
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_input).float().to(self.device).unsqueeze(0)
            q = self.net(x).cpu().numpy()[0]
        # mask invalid actions
        mask = np.full_like(q, -1e9)
        mask[valid_actions] = q[valid_actions]
        return int(np.argmax(mask))

    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        self.net.train()
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        ns = torch.from_numpy(ns).float().to(self.device)
        d = torch.from_numpy(d).float().to(self.device)

        qvals = self.net(s)
        q_a = qvals.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qnext = self.target(ns)
            qnext_max = qnext.max(dim=1)[0]
            target = r + (1.0 - d) * (self.gamma * qnext_max)
        loss = ((q_a - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

        # epsilon decay
        if self.eps > self.min_eps:
            self.eps = max(self.min_eps, self.eps * self.eps_decay)

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "target": self.target.state_dict(),
            "eps": self.eps,
        }, path)

    def load(self, path: str):
        if os.path.exists(path):
            data = torch.load(path, map_location=self.device)
            self.net.load_state_dict(data["net"])
            self.target.load_state_dict(data.get("target", data["net"]))
            self.eps = data.get("eps", self.eps)
            print("Loaded DQN weights:", path)

# -------------------- Training loop --------------------
def train_dqn_on_boards(training_data: List[Tuple[List[List[int]], List[Tuple[int,int]]]],
                        max_N: int,
                        max_color: int,
                        episodes_per_board: int = DEFAULT_EPISODES_PER_BOARD,
                        max_steps_mult: int = DEFAULT_MAX_STEPS_MULT,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        device: str = "cpu",
                        save_every_boards: int = 5,
                        boards_limit: int = None):
    agent = DQNAgent(max_N=max_N, max_color=max_color, device=device, batch_size=batch_size)
    weights_path = os.path.join(WEIGHTS_DIR, "dqn_unsupervised.pth")
    agent.load(weights_path)

    total = len(training_data) if boards_limit is None else min(len(training_data), boards_limit)
    solved_total = 0

    for idx, (board, _) in enumerate(training_data[:total], start=1):
        N = len(board)
        print(f"\n--- DQN training on board {idx}/{total} ({N}x{N}) ---")
        # reset per-board exploration to allow fresh exploration
        agent.eps = INITIAL_EPS
        best_progress = 0

        for ep in range(episodes_per_board):
            queens = []
            state = pad_and_encode(board, queens, max_N, max_color)
            episode_best = 0
            illegal_count = 0
            done_flag = False

            max_steps = max_steps_mult * N
            for step in range(max_steps):
                # prepare valid actions
                place_actions = [r * N + c for r in range(N) for c in range(N) if (r, c) not in queens]
                remove_action = N * N
                valid_actions = place_actions + [remove_action]

                a = agent.choose_action(state, valid_actions)

                # interpret remove-last
                if a == remove_action:
                    if not queens:
                        reward = REWARD_INVALID + STEP_PENALTY
                        ns = state
                        done = False
                        illegal_count += 1
                    else:
                        queens.pop()
                        reward = -0.5 + STEP_PENALTY
                        ns = pad_and_encode(board, queens, max_N, max_color)
                        done = False
                else:
                    # placement action may refer to an off-grid index if agent chooses padded area -> treat invalid
                    if a >= N * N:
                        reward = REWARD_INVALID + STEP_PENALTY
                        ns = state
                        done = False
                        illegal_count += 1
                    else:
                        r, c = divmod(int(a), N)
                        legal = is_legal_placement(board, queens, r, c)
                        if not legal:
                            reward = REWARD_INVALID + STEP_PENALTY
                            ns = state
                            done = False
                            illegal_count += 1
                        else:
                            queens.append((r, c))
                            progress = len(queens)
                            episode_best = max(episode_best, progress)
                            partial = progress / N
                            reward = REWARD_VALID + partial + STEP_PENALTY
                            ns = pad_and_encode(board, queens, max_N, max_color)
                            done = (progress == N)
                            if done:
                                reward += REWARD_COMPLETE
                                done_flag = True

                agent.push(state, a, reward, ns, float(done))
                agent.train_step()
                state = ns

                if done:
                    break

            if episode_best > best_progress:
                best_progress = episode_best

            # optional: if stuck for a long time without progress, slightly increase eps temporarily
            # (we keep this minimal in DQN; experience buffer + randomness usually suffice)

            if (ep % 200) == 0:
                print(f" ep={ep} best_progress={best_progress}/{N} buffer={len(agent.buffer)} eps={agent.eps:.4f}")

            # early stop if solved
            if best_progress == N:
                print(f"Board {idx} solved in ep {ep} (best_progress reached {best_progress}/{N})")
                solved_total += 1
                break

        # Save periodically
        if idx % save_every_boards == 0 or idx == total:
            agent.save(weights_path)
            print("Saved DQN weights to", weights_path)

    print(f"\nDQN training complete. solved {solved_total}/{total}")

# -------------------- CLI & run --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-per-board", type=int, default=DEFAULT_EPISODES_PER_BOARD)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    p.add_argument("--boards-limit", type=int, default=None, help="Limit number of boards to train (for debugging)")
    p.add_argument("--save-every", type=int, default=5, help="Save weights every N boards")
    return p.parse_args()

def choose_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg

if __name__ == "__main__":
    args = parse_args()
    device = choose_device(args.device)
    print("Metadata dir:", METADATA_DIR)
    training, max_N, max_color = load_training_data(METADATA_DIR)
    print(f"Loaded {len(training)} boards (curriculum sorted). max_N={max_N} max_color={max_color}")
    train_dqn_on_boards(training, max_N, max_color,
                        episodes_per_board=args.episodes_per_board,
                        batch_size=args.batch_size,
                        device=device,
                        save_every_boards=args.save_every,
                        boards_limit=args.boards_limit)
