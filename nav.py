import copy
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

valid_room_numbers = np.array([3, 6, 9, 12])
valid_room_weights = np.array([0.1, 0.2, 0.3, 0.4])
valid_actions = np.array([[0, 2], [0, -2], [-2, 0], [2, 0]])

"""
Doors between rooms are also described as a grid cell,
therefore if the maze looks like:

Room_A - door - Room_B
  |
 door
  |
Room_C - door - Room_D

its corresponding maze looks like:

WWWWW
W   W
W WWW
W   W
WWWWW

where W means a wall. Thus, actions
are of step size 2.
"""

IMG_FOLDER = "./img/"
DEVICE = "cuda"
BATCH_SIZE = 100
GAMMA = 0.95
MAX_N_STEPS = 20
NEGATIVE_REWARD = -0.75
GOAL_REWARD = 1
VALID_REWARD = -0.04
D, W, E, S, T = -1, 0, 1, 2, 3 # door, wall, emtpy, start, target

def generate_maze():
    maze = np.zeros((45, 45)) # (12-1)*2*2+1
    maze.fill(W)
    n_rooms = np.random.choice(valid_room_numbers, 1, p=valid_room_weights)
    curr_n_rooms = 1
    curr_pos = np.array([22,22])
    min_x = min_y = max_x = max_y = 22
    maze[curr_pos[0], curr_pos[1]] = E

    while curr_n_rooms < n_rooms:
        action = random.choice(valid_actions)
        curr_pos += action
        x, y = curr_pos
        if maze[x,y] == W:
            curr_n_rooms += 1
            maze[x,y] = E
            maze[x-action[0]//2, y-action[1]//2] = D # draw the door
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    center_x = (max_x+min_x)//2
    center_y = (max_y+min_y)//2
    return maze[center_x-11:center_x+12,center_y-11:center_y+12]

def generate_data(n_maps=20000):
    mazes = []
    for _ in tqdm(range(n_maps)):
        maze = generate_maze()
        mazes.append(maze)
    np.save("./nav", np.array(mazes), allow_pickle=True)  

def visualize_data(to=1):
    mazes = np.load("./nav.npy")
    for _ in tqdm(range(to)):
        maze = mazes[_-1]
        plt.imshow(maze, cmap=plt.cm.gray)
        plt.savefig(IMG_FOLDER+str(_)+".png")
    plt.close()

###############################################################################
#
# Double Deep-Q Leanring for navigation
#
###############################################################################

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = args
        self.pos = (self.pos+1) % self.capacity

    def sample(self, size):
        batch = random.sample(self.buffer, size)
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Nav(nn.Module):
    def __init__(self):
        super(Nav, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(3, 4, 7, padding=3),  # [:,3,W,H] -> [:,4,W,H]
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, padding=2),  # [:,4,W,H] -> [:,8,W,H]
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, padding=2), # [:,8,W,H] -> [:,16,W,H]
            Flatten(),
            nn.Linear(8464, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.state_emb = nn.Embedding(4, 4)
        self.state_emb.weight.data = torch.Tensor([
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        self.state_emb.requires_grad = False
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_target.requires_grad = False
        self.epsilon = .1 # for e-greedy exploration
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.opt = torch.optim.Adam(parameters, lr=1e-4)
        self.rb  = ReplayBuffer()

    def forward(self, state, use_Q_target=False):
        """
        params:
            m: the map [B, 1, H, W]
            origin: starting pos
            goal: end pos
        return:
            Q_sa: Q(s, a) [B, S, A]
        """
        state = self.state_emb(state)
        state = state.unsqueeze(1).permute(0, 4, 2, 3, 1).squeeze(-1)
        Q_sa = self.Q_target(state) if use_Q_target else self.Q(state)
        return Q_sa

    def make_tensor(self, x, is_long=0):
        if is_long:
            return torch.LongTensor(x).to(DEVICE)
        else:
            return torch.Tensor(x).to(DEVICE)

    def update(self):
        s, a, r, ns, d = self.rb.sample(BATCH_SIZE)
        d = d.astype('float')
        s, a, r, ns, d = map(self.make_tensor, [s, a, r, ns, d], [1, 1, 0, 1, 0])
        a  = a.unsqueeze(1)
        r  = r.unsqueeze(1)
        
        Q_sa = self.forward(s).gather(1, a)
        V_ns = self.forward(ns, True).max(1)[0].detach()
        # compute the TD error
        self.opt.zero_grad()
        loss = F.mse_loss(Q_sa, r + (1-d) * GAMMA * V_ns)
        loss.backward()
        self.opt.step()

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, path="./nav.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path="./nav.pt"):
        self.load_state_dict(torch.load(path))


def plot_trajectory(maze, trajectory, goal, iter=-1):
    rgb = np.zeros((*maze.shape, 3))
    rgb[maze == 0] = np.array([0,0,0]) # black wall
    rgb[maze == 1] = np.array([1,1,1]) # white path

    step = 0.5 / MAX_N_STEPS / 2
    base = 0.3
    prev_pos = trajectory[0]
    for i, pos in enumerate(trajectory[1:]):
        middle = (prev_pos+pos) // 2
        rgb[middle[0], middle[1]] = np.array([0,0,base])
        base += step
        rgb[pos[0], pos[1]] = np.array([0,0,base])
        base += step
        prev_pos = pos

    start = trajectory[0]
    rgb[start[0], start[1]] = np.array([0,1,0]) # green start
    if (trajectory[-1] == goal).all():
        rgb[goal[0], goal[1]] = np.array([1,1,0]) # yellow goal if reached
    else:
        rgb[goal[0], goal[1]] = np.array([1,0,0]) # red goal if not
    plt.imshow(rgb)
    plt.savefig((IMG_FOLDER+"traj_%d.png") % iter)
    plt.close()

def evaluate(mazes, nav):
    nav.eval()
    rewards = []
    for maze in mazes:
        # sample start and goal positions
        empty = np.stack(np.where(maze==E)).transpose()
        idx = np.random.choice(empty.shape[0], 2, False)
        pos, goal = empty[idx]

        state = maze.copy()
        state[state == D] = E # map doors to path
        state[pos[0], pos[1]] = S
        state[goal[0], goal[1]] = T
        step = 0
        epi_reward = 0
        done = False
        while not done and step < MAX_N_STEPS:
            step += 1
            state_tensor = torch.LongTensor(state).unsqueeze(0).to(DEVICE)
            Q_sa = nav(state_tensor)
            a    = Q_sa.argmax(1).cpu().item()
            next_pos = pos + valid_actions[a]
                
            if is_valid_step(state, pos, next_pos):
                state[next_pos[0], next_pos[1]] = S
                state[pos[0], pos[1]] = E
                if (next_pos == goal).all():
                    done = True
                    epi_reward += GOAL_REWARD
                else:
                    epi_reward += VALID_REWARD
                pos = next_pos
            else:
                epi_reward += NEGATIVE_REWARD
        rewards.append(epi_reward)

    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    nav.train()
    return mean_r, std_r
    
def plot(stats):
    xtr = stats['tr_i']
    ytr = stats['tr_r']
    xte = stats['te_i']
    yte = stats['te_r']
    plt.figure()
    plt.title("nav rewards during training")
    plt.plot(xtr, ytr, 'r', label="train reward")
    plt.plot(xte, yte, 'b', label="test reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(IMG_FOLDER+"nav_train_curve.png")
    plt.close()

def is_valid_step(maze, from_pos, to_pos):
    if max(to_pos) > 22:
        return False
    if min(to_pos) < 0:
        return False
    if maze[to_pos[0], to_pos[1]] == W:
        return False
    door = (from_pos+to_pos)//2
    if maze[door[0], door[1]] == W:
        return False
    return True

def train(n_epochs=10000):
    mazes = np.load("nav.npy")
    split = int(len(mazes)*0.99)
    #split = 1
    mazes, test_mazes = mazes[:split], mazes[split:]
    nav   = Nav().to(DEVICE)
    
    #pbar = tqdm(total=n_epochs*len(mazes))
    n_episodes  = 0
    best_reward = -np.inf

    plot_stats = {
        "tr_i": [],
        "te_i": [],
        "tr_r": [],
        "te_r": [],
    }
    np.random.seed(1)
    avg_r = 0
    avg_n = 0

    use_expert = True
    for epoch in range(n_epochs):
        np.random.shuffle(mazes)
        for maze in mazes:
            use_expert = (np.random.rand() < 0.5)
            n_episodes += 1
            #pbar.update(1)
            # sample start and goal positions
            empty = np.stack(np.where(maze==E)).transpose()
            idx = np.random.choice(empty.shape[0], 2, False)
            #idx = np.array([0,len(empty)-1])
            pos, goal = empty[idx]

            state = maze.copy()
            state[state == D] = E # map doors to path
            state[pos[0], pos[1]] = S
            state[goal[0], goal[1]] = T

            positions = [pos.copy()]
            states = [state.copy()]
            actions = []
            rewards = []
            dones = []
            if not use_expert:
                step = 0
                done = False
                while not done and step < MAX_N_STEPS:
                    a = 0
                    if np.random.rand() < nav.epsilon: # e-greedy
                        options = []
                        for _ in range(4):
                            next_pos = pos+valid_actions[_]
                            if is_valid_step(state, pos, next_pos):
                                options.append(_)
                        if len(options) == 0:
                            break
                        else:
                            a = random.choice(options)
                    else:
                        state_tensor = torch.LongTensor(state).unsqueeze(0).to(DEVICE)
                        Q_sa = nav(state_tensor)
                        a    = Q_sa.argmax(1).cpu().item()

                    next_pos = pos + valid_actions[a]
                        
                    if is_valid_step(state, pos, next_pos):
                        state[next_pos[0], next_pos[1]] = S
                        state[pos[0], pos[1]] = E
                        if (next_pos == goal).all():
                            done = True
                            rewards.append(GOAL_REWARD)
                        else:
                            #r = 1 - np.sum((next_pos-goal)**2) / np.sum((pos-goal)**2)
                            #rewards.append(r)
                            rewards.append(VALID_REWARD)
                        pos = next_pos
                    else:
                        rewards.append(NEGATIVE_REWARD)

                    positions.append(pos.copy())
                    states.append(state.copy())
                    actions.append(a)
                    dones.append(done)
                    step += 1
            else:
                # BFS expert trajectory
                queue = [pos]
                state[pos[0], pos[1]] = E
                visited = [pos[0]*23+pos[1]]
                parent_ptr  = [-1]
                transitions = [-1]
                idx = 0
                done = False
                while not done:
                    pos = queue[idx]
                    for i, action in enumerate(valid_actions):
                        possible_next_pos = pos+action
                        hash_code = possible_next_pos[0]*23 + possible_next_pos[1] 
                        if is_valid_step(state, pos, possible_next_pos) and hash_code not in visited:
                            queue.append(possible_next_pos)
                            parent_ptr.append(idx)
                            transitions.append(i)
                            visited.append(hash_code)
                            if (possible_next_pos == goal).all():
                                done = True
                                break
                    idx += 1
                idx = len(queue)-1
                trace = [idx]
                while parent_ptr[idx] != -1:
                    idx = parent_ptr[idx]
                    trace.insert(0, idx)

                for i in trace[1:]:
                    pos = queue[i]
                    maze_copy = state.copy()
                    maze_copy[goal[0], goal[1]] = T
                    maze_copy[pos[0], pos[1]] = S
                    positions.append(pos.copy())
                    states.append(maze_copy)
                    actions.append(transitions[i])
                    rewards.append(VALID_REWARD)
                    dones.append(False)
                rewards[-1] = GOAL_REWARD
                dones[-1] = True

            # perform HER, Alg 1 in https://arxiv.org/pdf/1707.01495.pdf
            states, next_states = states[:-1], states[1:]
            for i, (s, a, r, ns, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
                nav.rb.push(s, a, r, ns, d)
                if i < len(next_states)-2:
                    goals_ = random.sample(positions[i+1:], 2)
                    for g_ in goals_:
                        r_ = r
                        d_ = False
                        if (positions[i+1] == g_).all():
                            r_ = GOAL_REWARD
                            d_ = True
                        #elif r_ > NEGATIVE_REWARD:
                        #    r_ = 1 - np.sum((positions[i+1] - g_)**2) / np.sum((positions[i] - g_)**2)

                        s_, ns_ = s.copy(), ns.copy()
                        s_[goal[0], goal[1]] = E
                        ns_[goal[0], goal[1]] = E
                        s_[g_[0], g_[1]] = T
                        ns_[g_[0], g_[1]] = T
                        nav.rb.push(s_, a, r_, ns_, d_)
            
            if not use_expert:
                avg_r += np.sum(np.array(rewards))
                avg_n += 1

            if len(nav.rb) > BATCH_SIZE:
                # optimize nav policy (tabular Q)
                nav.update()
                # decrease epsilon
                if n_episodes >= 1000000 and (n_episodes - 1000000) % 1000 == 0:
                    nav.epsilon = max(nav.epsilon-1e-3, 0)

                if n_episodes % 100 == 0:
                    nav.update_target()
                    avg_r = avg_r/(avg_n+1e-12)
                    print("epi: %s | r: %s" % (n_episodes, avg_r))
                    plot_stats['tr_i'].append(n_episodes)
                    plot_stats['tr_r'].append(avg_r)
                    avg_r = avg_n = 0

                if n_episodes % 1000 == 0:
                    te_r, std_r = evaluate(test_mazes, nav)
                    plot_stats['te_i'].append(n_episodes)
                    plot_stats['te_r'].append(te_r)
                    plot(plot_stats)
                    if te_r > best_reward:
                        print("[INFO] found best r %s, std %s, saving ..." % (te_r, std_r))
                        nav.save()
                        best_reward = te_r

                if n_episodes % 1000 == 0:
                    m = maze.copy()
                    m[m == D] = E
                    plot_trajectory(m, positions, goal, n_episodes)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'g':
            generate_data()
        elif sys.argv[1] == 'v':
            visualize_data()
        else:
            print("[ERROR] unknown flag ('g' for generate data, 'v' for visualize data)")
    else:
        train()
