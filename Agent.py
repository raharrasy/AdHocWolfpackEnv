import random
import math
import sys
import queue as Q
from ReplayMemory import ReplayMemoryLite, ReplayMemory
from QNetwork import DQN, AdHocWolfpackGNN, GraphOppoModel, MADDPGDQN
from misc import hard_copy, soft_copy
from MADDPGMisc import gumbel_softmax
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import torch.distributions as dist
#import ray


class Agent(object):
    def __init__(self, agent_id, obs_type):
        self.agent_id = agent_id
        self.obs_type = obs_type

    def get_obstype(self):
        return self.obs_type


class RandomAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(RandomAgent, self).__init__(agent_id, obs_type)

    def act(self, obs=None):
        return random.randint(0, 6)


class GreedyPredatorAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyPredatorAgent, self).__init__(agent_id, obs_type)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        pounce_res = self.pounce_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats)
        if pounce_res != -1:
            return pounce_res
        approach_res = self.approach_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, poss_locs)
        if approach_res != -1:
            return approach_res
        else:
            return random.randint(0, 6)

    def pounce_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats):
        next_to_oppo = [self.computeManhattanDistance(agent_pos, oppo_next) < 2 for
                        idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]
        next_to_oppo_idx = [idx for
                            idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]

        if not any(next_to_oppo):
            return -1
        else:
            point1 = agent_pos
            point2 = oppo_pos[next_to_oppo_idx[next_to_oppo.index(True)]]
            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            compass_dir = 0
            if x_del > 0:
                compass_dir = 3
            elif x_del < 0:
                compass_dir = 1
            elif y_del < 0:
                compass_dir = 2

            real_dir = (compass_dir - agent_orientation) % 4
            return real_dir

    def approach_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, possible_positions):
        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(pos[0] + adder[0], pos[1] + adder[1])
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1]) in possible_positions]

        if len(boundary_points) != 0:
            manhattan_distances = [self.computeManhattanDistance(agent_pos, boundary_point)
                                   for boundary_point in boundary_points]
            min_idx = manhattan_distances.index(min(manhattan_distances))
            point1 = agent_pos
            point2 = boundary_points[min_idx]

            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            next_point_x = [point1[0], point1[1]]
            next_point_y = [point1[0], point1[1]]

            compass_dir_y = 0
            compass_dir_x = 1

            if y_del < 0:
                compass_dir_y = 2

            if x_del > 0:
                compass_dir_x = 3

            if compass_dir_y == 0:
                next_point_y[0] = next_point_y[0] - 1
            else:
                next_point_y[0] = next_point_y[0] + 1

            if compass_dir_x == 1:
                next_point_x[1] = next_point_x[1] + 1
            else:
                next_point_x[1] = next_point_x[1] - 1

            available_flags = [tuple(next_point_y) in possible_positions, tuple(next_point_x) in possible_positions]

            if not any(available_flags):
                return random.randint(0, 3)
            elif all(available_flags):
                if abs(y_del) > abs(x_del):
                    return (compass_dir_y - agent_orientation) % 4
                else:
                    return (compass_dir_x - agent_orientation) % 4
            else:
                if available_flags[0]:
                    return (compass_dir_y - agent_orientation) % 4

            return (compass_dir_x - agent_orientation) % 4
        return -1

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class GreedyProbabilisticAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyProbabilisticAgent, self).__init__(agent_id, obs_type)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        pounce_res = self.pounce_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats)
        if pounce_res != -1:
            return pounce_res
        approach_res = self.approach_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, poss_locs)
        if approach_res != -1:
            return approach_res
        else:
            return random.randint(0, 6)

    def pounce_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats):
        next_to_oppo = [self.computeManhattanDistance(agent_pos, oppo_next) < 2 for
                        idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]
        next_to_oppo_idx = [idx for
                            idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]

        if not any(next_to_oppo):
            return -1
        else:
            point1 = agent_pos
            point2 = oppo_pos[next_to_oppo_idx[next_to_oppo.index(True)]]
            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            compass_dir = 0
            if x_del > 0:
                compass_dir = 3
            elif x_del < 0:
                compass_dir = 1
            elif y_del < 0:
                compass_dir = 2

            real_dir = (compass_dir - agent_orientation) % 4
            return real_dir

    def approach_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, possible_positions):
        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(pos[0] + adder[0], pos[1] + adder[1])
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1]) in possible_positions]

        if len(boundary_points) != 0:
            manhattan_distances = [self.computeManhattanDistance(agent_pos, boundary_point)
                                   for boundary_point in boundary_points]
            min_idx = manhattan_distances.index(min(manhattan_distances))
            point1 = agent_pos
            point2 = boundary_points[min_idx]

            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]
            subtractor = max(abs(y_del), abs(x_del))
            y_prob = math.exp((abs(y_del) - subtractor) / 2.5) / (
                        math.exp((abs(y_del) - subtractor) / 2.5) + math.exp((abs(x_del) - subtractor) / 2.5))

            dim = 0
            if random.random() > y_prob:
                dim = 1

            v_opt = abs(y_del) + abs(x_del) - 1
            v_less_opt = abs(y_del) + abs(x_del) + 1

            opt_dest = None
            sub_opt_dest = None
            compass_opt = None
            if dim == 0:
                if y_del > 0:
                    opt_dest = (point1[0] - 1, point1[1])
                    sub_opt_dest = (point1[0] + 1, point1[1])
                    compass_opt = 0
                else:
                    opt_dest = (point1[0] + 1, point1[1])
                    sub_opt_dest = (point1[0] - 1, point1[1])
                    compass_opt = 2
            else:
                if x_del > 0:
                    opt_dest = (point1[0], point1[1] - 1)
                    sub_opt_dest = (point1[0], point1[1] + 1)
                    compass_opt = 3
                else:
                    opt_dest = (point1[0], point1[1] + 1)
                    sub_opt_dest = (point1[0], point1[1] - 1)
                    compass_opt = 1

            if not opt_dest in possible_positions:
                v_opt += 5

            if not sub_opt_dest in possible_positions:
                v_less_opt += 5

            subtractor = max(v_opt, v_less_opt)
            opt_prob = (math.exp((abs(v_opt) - subtractor) / -2.5)) / (
                        math.exp((abs(v_opt) - subtractor) / -2.5) + (math.exp((abs(v_less_opt) - subtractor) / -2.5)))

            dir = compass_opt
            if random.random() > opt_prob:
                dir = (compass_opt + 2) % 4

            real_dir = (dir - agent_orientation) % 4
            return real_dir

        return -1

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class TeammateAwarePredator(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(TeammateAwarePredator, self).__init__(agent_id, obs_type)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        all_agent_pos = obs[0]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        adders = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        border_points = [[((point[0] + adder[0], point[1] + adder[1]), point) for adder in adders
                          if (point[0] + adder[0], point[1] + adder[1]) in poss_locs and
                          (point[0] + adder[0], point[1] + adder[1]) not in all_agent_pos] for point in oppo_pos]
        collapsed_border = [point for elem in border_points for point in elem]
        manhattan_dists = [list(zip(collapsed_border,
                                    [self.computeManhattanDistance(agent, elem_border[0]) for elem_border in
                                     collapsed_border]))
                           for agent in all_agent_pos]

        max_manhattan_dists = [max(k) for k in manhattan_dists]
        enumerated_man_dist = list(enumerate(max_manhattan_dists))
        enumerated_man_dist.sort(key=(lambda x: x[1]), reverse=True)

        for a in manhattan_dists:
            a.sort(key=(lambda x: x[1]))

        dests = []
        oppo_chased = []
        for b in enumerated_man_dist:
            a = manhattan_dists[b[0]]
            idx = 0
            while idx < len(a) and a[idx][0][0] in dests:
                idx += 1

            if idx < len(a):
                dests.append(a[idx][0][0])
                oppo_chased.append(a[idx][0][1])
            else:
                dests.append(None)
                oppo_chased.append(None)

        single_dest = dests[self.agent_id]
        single_oppo_chased = oppo_chased[self.agent_id]

        if single_oppo_chased == None:
            return random.randint(0, 6)

        point1 = agent_pos
        point2 = single_oppo_chased
        y_del = point1[0] - point2[0]
        x_del = point1[1] - point2[1]

        compass_dir = -1
        if x_del == 1:
            compass_dir = 3
        elif x_del == -1:
            compass_dir = 1
        elif y_del == -1:
            compass_dir = 2
        elif y_del == 1:
            compass_dir = 0

        if compass_dir != -1:
            real_dir = (compass_dir - agent_orientation) % 4
            return real_dir

        next_dest = None
        start = agent_pos
        if single_dest != 0:
            end = single_dest
            next_dest = self.a_star(start, end, poss_locs, all_agent_pos, oppo_pos)
        else:
            return random.randint(0, 6)

        if next_dest == None:
            return random.randint(0, 6)
        point1 = agent_pos
        point2 = next_dest
        y_del = point1[0] - point2[0]
        x_del = point1[1] - point2[1]

        compass_dir = 0
        if x_del > 0:
            compass_dir = 3
        elif x_del < 0:
            compass_dir = 1
        elif y_del < 0:
            compass_dir = 2

        real_dir = (compass_dir - agent_orientation) % 4
        return real_dir

    def a_star(self, start, end, possible_states, teammate_locs, oppo_locs):
        frontier = Q.PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        idx = 0
        while not frontier.empty():
            current = frontier.get()
            if current == end:
                curr_end = current
                prev = came_from[curr_end]

                while prev != start:
                    curr_end = prev
                    prev = came_from[curr_end]
                return curr_end

            for next in [(current[0] + adder[0], current[1] + adder[1]) for adder in adders if
                         (current[0] + adder[0], current[1] + adder[1]) in possible_states
                         and (current[0] + adder[0], current[1] + adder[1]) not in teammate_locs
                         and (current[0] + adder[0], current[1] + adder[1]) not in oppo_locs]:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.computeManhattanDistance(next, end)
                    frontier.put(next, priority)
                    came_from[next] = current

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class DQNAgent(Agent):
    def __init__(self, agent_id, args=None, obs_type="full_rgb", obs_height=9, obs_width=17, mode="test"):
        super(DQNAgent, self).__init__(agent_id, obs_type)
        self.obs_type = obs_type
        self.args = args
        self.experience_replay = ReplayMemoryLite(state_h=obs_height, state_w=obs_width,
                                                  with_gpu=self.args['with_gpu'])
        self.dqn_net = DQN(17,9,32,self.args['max_seq_length'],7, mode="partial")

        if self.args['with_gpu']:
            self.dqn_net.cuda()
            self.dqn_net.device = "cuda:0"
            self.target_dqn_net.cuda()
            self.target_dqn_net.device = "cuda:0"

        self.mode = mode
        if not self.mode == "test":
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
            self.target_dqn_net = DQN(17, 9, 32, self.args['max_seq_length'], 7, mode="partial")
            hard_copy(self.target_dqn_net, self.dqn_net)

        self.recent_obs_storage = np.zeros([self.args['max_seq_length'], obs_height, obs_width, 3])


    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def act(self, obs,added_features=None, mode="train", epsilon=0.01):
        self.recent_obs_storage = np.roll(self.recent_obs_storage, axis=0, shift=-1)
        self.recent_obs_storage[-1] = obs
        net_inp = torch.Tensor([self.recent_obs_storage.transpose([0, 3, 1, 2])])
        _, indices = torch.max(self.dqn_net(net_inp), dim=-1)
        # Implement resets
        if not self.mode=="test":
            if random.random() < epsilon:
                indices = random.randint(0,6)
        return indices

    def store_exp(self, exp):
        self.experience_replay.insert(exp)

    def get_obs_type(self):
        return self.obs_type

    def update(self):
        if self.experience_replay.size < self.args['sampling_wait_time']:
            return
        batched_data = self.experience_replay.sample(self.args['batch_size'])
        state, action, reward, dones, next_states = batched_data[0], batched_data[1], batched_data[2], \
                                                    batched_data[3], batched_data[4]

        state = state.permute(0, 1, 4, 2, 3)
        next_states = next_states.permute(0, 1, 4, 2, 3)

        predicted_value = self.dqn_net(state).gather(1, action.long())
        target_values = reward + self.args['disc_rate'] * (1 - dones) * torch.max(self.target_dqn_net(next_states),
                                                                                  dim=-1, keepdim=True)[0]
        loss = 0.5 * torch.mean((predicted_value - target_values.detach()) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net)

class DistilledCoopAStarAgent(object):
    def __init__(self, id, obs_type="full_rgb_graph"):
        self.agent_id = id
        self.obs_type = obs_type
        self.dqn_net = GraphOppoModel(6, 0, 50, 4, 40, 20, 30, 10, 15, 7)
        self.load_params("assets/distilled_teamwork_net/distilled_net.pkl")

    def load_params(self, dir):
        self.dqn_net.load_state_dict(torch.load(dir))

    def act(self, obs):
        image_obs = torch.Tensor([obs[0]]).permute(0,3,1,2)
        node_obs = torch.Tensor(obs[1])
        added_oppo_inf = []
        for pos_tuples in obs[2]:
            list_pos = list(pos_tuples)
            added_oppo_inf.extend(list_pos)
        added_tensor = torch.Tensor([added_oppo_inf])

        graph = dgl.DGLGraph()
        num_nodes = len(obs[1])
        graph.add_nodes(num_nodes)
        src, dst = tuple(zip(*[(i,j) for i in range(num_nodes) for j in range(num_nodes) if i != j]))
        graph.add_edges(src, dst)
        fin_graph = dgl.batch([graph])
        edge_feats = torch.zeros([graph.number_of_edges(),0])

        logits_all = self.dqn_net(fin_graph, edge_feats, node_obs, image_obs, added_tensor)

        m = dist.Categorical(logits=logits_all[0])
        act = m.sample()

        return act

class MADDPGAgent(Agent):
    def __init__(self, agent_id, max_seq_length=10, obs_type="full_rgb", with_gpu=False, obs_height=25, obs_width=25,
                 mode="test"):
        super(MADDPGAgent, self).__init__(agent_id, obs_type)
        self.obs_type = obs_type
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.with_gpu = with_gpu

        self.dqn_net = MADDPGDQN(25,25,32,self.max_seq_length,7, mode="full",
                      extended_feature_len = 6)
        self.load_params()

        if not self.mode == "test":
            self.experience_replay = ReplayMemoryLite(state_h=obs_height, state_w=obs_width,
                                                      with_gpu=self.with_gpu)

        if self.with_gpu:
            self.dqn_net.cuda()
            self.dqn_net.device = "cuda:0"
            self.target_dqn_net.cuda()
            self.target_dqn_net.device = "cuda:0"

        self.mode = mode
        if not self.mode == "test":
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
            self.target_dqn_net = DQN(25, 25, 32, self.max_seq_length, 7, mode="full",
                                      extended_feature_len=6)
            hard_copy(self.target_dqn_net, self.dqn_net)

        self.recent_obs_storage = np.zeros([self.max_seq_length, obs_height, obs_width, 3])
        self.added_obs_storage = np.zeros([self.max_seq_length, 6])

    def load_params(self):
        random_agent_id = random.randint(0,2)
        filename = 'assets/maddpg_agent_parameters/Agent_'+str(random_agent_id)+'_204.pkl'
        self.dqn_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.dqn_net.eval()

    def step(self, model, observation, add_obs, temperature=1.0, hard=False, epsilon=0.0, evaluate=False):
        logits = model(observation, add_obs)
        output = gumbel_softmax(logits, temperature, hard, epsilon, evaluate)
        return logits, output

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def act(self, obs,added_features=None, mode="train", epsilon=0.01):
        self.recent_obs_storage = np.roll(self.recent_obs_storage, axis=0, shift=-1)
        self.added_obs_storage = np.roll(self.added_obs_storage, axis=0, shift=-1)
        self.recent_obs_storage[-1] = obs[0]
        self.added_obs_storage[-1] = np.asarray(obs[1])

        action_tensor = self.step(self.dqn_net, torch.Tensor([self.recent_obs_storage.transpose([0, 3, 1, 2])]),
                                  torch.Tensor([self.added_obs_storage]),
                               hard=True, epsilon=epsilon, evaluate=True)

        action = action_tensor[1][0].detach().argmax().item()

        return action

    def get_obs_type(self):
        return self.obs_type

class AdHocLearningAgent(Agent):
    def __init__(self, agent_id=0, args=None, obs_type="adhoc_obs", rollout_freq=8, back_prop_len=12, optimizer=None,
                 mode="train", device=None):
        super(AdHocLearningAgent, self).__init__(agent_id=agent_id, obs_type=obs_type)
        self.args = args

        # Initialize neural network dimensions
        self.dim_lstm_out = 10
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dqn_net = AdHocWolfpackGNN(3, 0, 20, 40, 20, 30, 15,
                                        10, 7, with_rfm = False).to(self.device)
        self.target_dqn_net = AdHocWolfpackGNN(3, 0, 20, 40, 20, 30, 15,
                                        10, 7, with_rfm = False).to(self.device)
        hard_copy(self.target_dqn_net,  self.dqn_net)
        self.mode = mode

        # Initialize hidden states of prediction
        self.hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.hidden_node = [None for _ in range(self.args['num_envs'])]
        self.hidden_u = [None for _ in range(self.args['num_envs'])]

        # Initialize hidden states of target
        self.target_hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_node = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_u = [None for _ in range(self.args['num_envs'])]

        # Set params for Ad Hoc BPTT
        self.rollout_freq = rollout_freq
        self.back_prop_len = back_prop_len
        self.retain_comp_graph = self.back_prop_len > self.rollout_freq
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
        self.hidden_hist = []
        self.prev_hidden = None
        self.curr_hidden = None

        # Set params for step
        self.graph = [None for _ in range(self.args['num_envs'])]
        self.next_graph = [None for _ in range(self.args['num_envs'])]
        self.obs = None
        self.next_obs = None

        # Stored data for training
        self.predicted_vals = None
        self.pred_val_pointer = None
        self.target_vals = None

        self.loss_module = nn.MSELoss()

    def step(self, obs):
        if self.next_obs is None:
            outs = self.prep_obs(obs, self.hidden_edge, self.hidden_node, self.hidden_u)
            prepped_obs = outs
            self.graph = outs[0]

            #Initialize target values
            target_obs = self.prep_obs(obs, self.target_hidden_edge,
                                            self.target_hidden_node, self.target_hidden_u)
            target_batch_graph = dgl.batch(target_obs[0])
            _, t_e_hid, t_n_hid, t_u_hid = self.target_dqn_net(target_batch_graph, target_obs[1], target_obs[2],
                                                               target_obs[3], target_obs[4], target_obs[5],
                                                               target_obs[6])

            self.target_hidden_edge = t_e_hid
            self.target_hidden_node = t_n_hid
            self.target_hidden_u = list(
                zip([hid[None, None, :] for hid in t_u_hid[0][0]], [hid[None, None, :] for hid in t_u_hid[1][0]]))
        else:
            self.graph, prepped_obs = self.next_graph, self.next_obs

        self.obs = prepped_obs

        if len(self.hidden_hist) == 0:
            self.prev_hidden = None
            a, b = self.obs[4][0].detach(), self.obs[4][1].detach()
            c, d = self.obs[5][0].detach(), self.obs[5][1].detach()
            e, f = self.obs[6][0].detach(), self.obs[6][1].detach()
            a.requires_grad = True
            b.requires_grad = True
            c.requires_grad = True
            d.requires_grad = True
            e.requires_grad = True
            f.requires_grad = True
            p_hid_e = (a, b)
            p_hid_n = (c, d)
            p_hid_u = (e, f)
            self.curr_hidden = (p_hid_e, p_hid_n, p_hid_u)
            self.hidden_hist.append((None,(self.obs[4], self.obs[5], self.obs[6])))

        batch_graph = dgl.batch(self.obs[0])
        #prev hidden cuyr hidden?


        out, e_hid, n_hid, u_hid = self.dqn_net(batch_graph,self.obs[1], self.obs[2], self.obs[3],
                                                        self.curr_hidden[0], self.curr_hidden[1],
                                                self.curr_hidden[2])

        self.hidden_edge = e_hid
        self.hidden_node = n_hid
        self.hidden_u = list(zip([hid[None,None,:] for hid in u_hid[0][0]], [hid[None,None,:] for hid in u_hid[1][0]]))

        act = torch.argmax(out, dim=-1)
        self.predicted_vals.append(out.gather(1, act[:,None]))
        act = [a.item() for a in act]

        return act

    def set_next_state(self, next_obs, rewards, dones):
        # Set all necessary data for next forward computation
        self.next_obs = self.prep_obs(next_obs, self.hidden_edge, self.hidden_node, self.hidden_u)
        self.next_graph = self.next_obs[0]
        self.prev_hidden = self.curr_hidden

        a, b = self.next_obs[4][0].detach(), self.next_obs[4][1].detach()
        c, d = self.next_obs[5][0].detach(), self.next_obs[5][1].detach()
        e, f = self.next_obs[6][0].detach(), self.next_obs[6][1].detach()
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        d.requires_grad = True
        e.requires_grad = True
        f.requires_grad = True
        p_hid_e = (a, b)
        p_hid_n = (c, d)
        p_hid_u = (e, f)

        self.curr_hidden = (p_hid_e, p_hid_n, p_hid_u)

        self.hidden_hist.append((self.prev_hidden, (self.next_obs[4], self.next_obs[5], self.next_obs[6])))
        if len(self.hidden_hist) > self.back_prop_len:
            del self.hidden_hist[0]

        # Compute target values
        # Initialize target values
        target_obs = self.prep_obs(next_obs, self.target_hidden_edge,
                                       self.target_hidden_node, self.target_hidden_u)
        target_batch_graph = dgl.batch(target_obs[0])
        targ_out, t_e_hid, t_n_hid, t_u_hid = self.target_dqn_net(target_batch_graph, target_obs[1], target_obs[2],
                                                               target_obs[3], target_obs[4], target_obs[5],
                                                               target_obs[6])
        targs = torch.max(targ_out, dim=-1)[0][:,None]
        rewards = torch.Tensor(rewards)[:,None].to(self.device)
        dones = torch.Tensor(dones)[:,None].to(self.device)

        targs = rewards + self.args['disc_rate'] * (1-dones) * targs
        self.target_vals.append(targs)

        self.target_hidden_edge = t_e_hid
        self.target_hidden_node = t_n_hid
        self.target_hidden_u = list(zip([hid[None, None, :] for hid in t_u_hid[0][0]],
                                        [hid[None, None, :] for hid in t_u_hid[1][0]]))

    def prep_obs(self, obses, prev_hidden_e, prev_hidden_n, prev_hidden_u):
        # All the list used for creating batches
        new_graphs = []
        edge_feature_list = []
        node_feature_list = []
        u_feature_list = []
        prep_hidden_e_list = []
        prep_hidden_n_list = []

        for idx, obs in enumerate(obses):
            if prev_hidden_e[idx] is None:
                prev_hidden_e[idx] = (torch.zeros([1, len(obs[2]) * (len(obs[2]) - 1), 10]).to(self.device),
                                         torch.zeros([1, len(obs[2]) * (len(obs[2]) - 1), 10]).to(self.device))
            if prev_hidden_n[idx] is None:
                prev_hidden_n[idx] = (torch.zeros([1, len(obs[2]), 10]).to(self.device), torch.zeros([1, len(obs[2]),
                                                                                                      10]).to(self.device))
            if prev_hidden_u[idx] is None:
                prev_hidden_u[idx] = (torch.zeros([1, 1, 10]).to(self.device), torch.zeros([1, 1, 10]).to(self.device))

            new_graph, edge_filters = self.create_input_graph(obs[2], obs[3], idx)
            new_graphs.append(new_graph)

            # Calculate number of added nodes and new number of nodes
            added_n, new_node_num = obs[3], new_graph.nodes().shape[0]
            # Calculate number of nodes after filtering
            after_delete_node = new_node_num - added_n
            # Calculate the added number of edges
            added_e = (new_node_num * (new_node_num - 1)) - (after_delete_node * (after_delete_node - 1))

            # Prepare hiddens

            # Empty features for edge
            edge_features = torch.Tensor(size=[new_node_num * (new_node_num - 1), 0]).to(self.device)
            edge_feature_list.append(edge_features)
            # Features for nodes
            node_features = torch.Tensor(obs[0]).to(self.device)
            node_feature_list.append(node_features)
            # Use image as features for graph
            u_features = torch.Tensor(obs[1]).permute(2, 0, 1)[None, :, :, :].to(self.device)
            u_feature_list.append(u_features)

            preprocessed_hidden_e = self.prep_hidden(prev_hidden_e[idx], [edge_filters], [added_e])
            prep_hidden_e_list.append(preprocessed_hidden_e)
            preprocessed_hidden_n = self.prep_hidden(prev_hidden_n[idx], [torch.Tensor(obs[2]).to(self.device)],
                                                     [added_n])
            prep_hidden_n_list.append(preprocessed_hidden_n)

        e_feat, n_feat, u_feat = torch.cat(edge_feature_list, dim=0), \
                                 torch.cat(node_feature_list, dim=0), \
                                 torch.cat(u_feature_list, dim=0)

        hid_1_e, hid_2_e = zip(*prep_hidden_e_list)
        hid_e = (torch.cat(hid_1_e, dim=1), torch.cat(hid_2_e, dim=1))

        hid_1_n, hid_2_n = zip(*prep_hidden_n_list)
        hid_n = (torch.cat(hid_1_n, dim=1), torch.cat(hid_2_n, dim=1))

        hid_1_u, hid_2_u = zip(*self.hidden_u)
        hid_u = (torch.cat(hid_1_u, dim=1), torch.cat(hid_2_u, dim=1))

        return new_graphs, e_feat, n_feat, u_feat,\
               hid_e, hid_n, hid_u



    def reset(self, obs):
        self.hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.hidden_node = [None for _ in range(self.args['num_envs'])]
        self.hidden_u = [None for _ in range(self.args['num_envs'])]
        self.graph = [None for _ in range(self.args['num_envs'])]
        self.next_graph = [None for _ in range(self.args['num_envs'])]
        self.obs = None
        self.next_obs = None

        self.hidden_hist = []
        self.prev_hidden = None
        self.curr_hidden = None

        self.predicted_vals = []
        self.target_vals = []

        self.target_hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_node = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_u = [None for _ in range(self.args['num_envs'])]

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def create_input_graph(self, node_filters, num_added_nodes, idx=0):
        device = torch.device('cpu') if self.device == "cpu" else torch.device('cuda:0')
        new_graph = dgl.DGLGraph()
        new_graph.add_nodes(len(node_filters))
        src, dest = tuple(zip(*[(i, j) for i in range(len(node_filters)) for j in range(len(node_filters)) if i != j]))
        src_past, dest_past = tuple(zip(*[(i, j) for i in node_filters for j in node_filters if
                                          i != j]))
        new_graph.add_edges(src, dest)

        # Compute edge filters based on added and removed nodes
        if self.graph[idx] is None:
            edge_filters = new_graph.edge_ids(src_past, dest_past).long().to(self.device)
        else:
            edge_filters = self.graph[idx].edge_ids(src_past, dest_past).long().to(self.device)

        node_filters = torch.Tensor(node_filters).long().to(self.device)
        if num_added_nodes != 0:
            added = num_added_nodes
            new_graph.add_nodes(num_added_nodes)
            new_idx = len(node_filters)
            src, dest = tuple(zip(*[(i, j) for i in range(new_idx + added) for j in range(new_idx + added)
                                    if (i in range(new_idx, new_idx + added) or j in range(new_idx, new_idx + added))
                                    and i != j]))

            new_graph.add_edges(src, dest)
        new_graph.to(device)

        return new_graph, edge_filters

    def prep_hidden(self,hidden, remain_mask, added_num):

        hid_filtered = [(hidden[0].gather(1, remain_mask[0].long()[None, :, None].repeat(1, 1, self.dim_lstm_out)),
                           hidden[1].gather(1, remain_mask[0].long()[None, :, None].repeat(1, 1, self.dim_lstm_out)))]

        hid_added = [(torch.cat([k[0], torch.zeros([1, added, self.dim_lstm_out]).to(self.device)], dim=1),
                        torch.cat([k[1], torch.zeros([1, added, self.dim_lstm_out]).to(self.device)], dim=1))
                       for k, added in zip(hid_filtered, added_num)]

        e_hid_1, e_hid_2 = zip(*hid_added)
        hidden_processed = (torch.cat(e_hid_1, dim=1), torch.cat(e_hid_2, dim=1))

        return hidden_processed

    def AdHocBPTT(self):
        self.optimizer.zero_grad()
        # backprop last module (keep graph only if they ever overlap)
        for j in range(self.back_prop_len - 1):

            if j < len(self.predicted_vals):
                loss = self.loss_module(self.predicted_vals[-j - 1], self.target_vals[-j - 1].detach())
                loss.backward(retain_graph=True)

            # if we get all the way back to the "init_state", stop
            if self.hidden_hist[-j - 2][0] is None:
                break
            curr_grad_u = (self.hidden_hist[-j - 1][0][2][0]._grad, self.hidden_hist[-j - 1][0][2][1]._grad)
            self.hidden_hist[-j - 2][1][2][0].backward(curr_grad_u[0],
                                                       retain_graph=True)
            self.hidden_hist[-j - 2][1][2][1].backward(curr_grad_u[1],
                                                        retain_graph=True)

            curr_grad_n = (self.hidden_hist[-j - 1][0][1][0].grad, self.hidden_hist[-j - 1][0][1][1].grad)
            self.hidden_hist[-j - 2][1][1][0].backward(curr_grad_n[0],
                                                     retain_graph=True)
            self.hidden_hist[-j - 2][1][1][1].backward(curr_grad_n[1],
                                                        retain_graph=True)

            curr_grad_e = (self.hidden_hist[-j - 1][0][0][0].grad, self.hidden_hist[-j - 1][0][0][1].grad)
            self.hidden_hist[-j - 2][1][0][0].backward(curr_grad_e[0],
                                                     retain_graph=True)
            self.hidden_hist[-j - 2][1][0][1].backward(curr_grad_e[1],
                                                        retain_graph=True)

        self.optimizer.step()
        soft_copy(self.target_dqn_net, self.dqn_net, self.args['tau'])
        self.predicted_vals = []
        self.target_vals = []

class AdHocShortBPTTAgent(Agent):
    def __init__(self, agent_id=0, args=None, obs_type="adhoc_obs", rollout_freq=8, back_prop_len=12, optimizer=None,
                 mode="train", device=None):
        super(AdHocShortBPTTAgent, self).__init__(agent_id=agent_id, obs_type=obs_type)
        self.args = args

        # Initialize neural network dimensions
        self.dim_lstm_out = 10
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dqn_net = AdHocWolfpackGNN(3, 0, 20, 40, 20, 30, 15,
                                        10, 7, with_rfm = False).to(self.device)
        self.target_dqn_net = AdHocWolfpackGNN(3, 0, 20, 40, 20, 30, 15,
                                        10, 7, with_rfm = False).to(self.device)
        hard_copy(self.target_dqn_net,  self.dqn_net)
        self.mode = mode

        # Initialize hidden states of prediction
        self.hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.hidden_node = [None for _ in range(self.args['num_envs'])]
        self.hidden_u = [None for _ in range(self.args['num_envs'])]

        # Initialize hidden states of target
        self.target_hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_node = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_u = [None for _ in range(self.args['num_envs'])]

        # Set params for Ad Hoc BPTT
        self.rollout_freq = rollout_freq
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
        self.prev_hidden = None
        self.curr_hidden = None

        # Set params for step
        self.graph = [None for _ in range(self.args['num_envs'])]
        self.next_graph = [None for _ in range(self.args['num_envs'])]
        self.obs = None
        self.next_obs = None

        # Stored data for training
        self.predicted_vals = None
        self.pred_val_pointer = None
        self.target_vals = None

        self.loss_module = nn.MSELoss()

    def step(self, obs):
        if self.next_obs is None:
            outs = self.prep_obs(obs, self.hidden_edge, self.hidden_node, self.hidden_u)
            prepped_obs = outs
            self.graph = outs[0]

            #Initialize target values
            target_obs = self.prep_obs(obs, self.target_hidden_edge,
                                            self.target_hidden_node, self.target_hidden_u)
            target_batch_graph = dgl.batch(target_obs[0])
            _, t_e_hid, t_n_hid, t_u_hid = self.target_dqn_net(target_batch_graph, target_obs[1], target_obs[2],
                                                               target_obs[3], target_obs[4], target_obs[5],
                                                               target_obs[6])

            self.target_hidden_edge = t_e_hid
            self.target_hidden_node = t_n_hid
            self.target_hidden_u = list(
                zip([hid[None, None, :] for hid in t_u_hid[0][0]], [hid[None, None, :] for hid in t_u_hid[1][0]]))
        else:
            self.graph, prepped_obs = self.next_graph, self.next_obs

        self.obs = prepped_obs

        if self.prev_hidden is None:
            self.curr_hidden = (self.obs[4], self.obs[5], self.obs[6])

        batch_graph = dgl.batch(self.obs[0])
        out, e_hid, n_hid, u_hid = self.dqn_net(batch_graph,self.obs[1], self.obs[2], self.obs[3],
                                                        self.curr_hidden[0], self.curr_hidden[1],
                                                self.curr_hidden[2])

        self.hidden_edge = e_hid
        self.hidden_node = n_hid
        self.hidden_u = list(zip([hid[None,None,:] for hid in u_hid[0][0]], [hid[None,None,:] for hid in u_hid[1][0]]))

        act = torch.argmax(out, dim=-1)
        self.predicted_vals.append(out.gather(1, act[:,None]))
        act = [a.item() for a in act]

        return act

    def set_next_state(self, next_obs, rewards, dones):
        # Set all necessary data for next forward computation
        self.next_obs = self.prep_obs(next_obs, self.hidden_edge, self.hidden_node, self.hidden_u)
        self.next_graph = self.next_obs[0]
        self.prev_hidden = self.curr_hidden

        self.curr_hidden = (self.next_obs[4], self.next_obs[5], self.next_obs[6])

        # Compute target values
        # Initialize target values
        target_obs = self.prep_obs(next_obs, self.target_hidden_edge,
                                       self.target_hidden_node, self.target_hidden_u)
        target_batch_graph = dgl.batch(target_obs[0])
        targ_out, t_e_hid, t_n_hid, t_u_hid = self.target_dqn_net(target_batch_graph, target_obs[1], target_obs[2],
                                                               target_obs[3], target_obs[4], target_obs[5],
                                                               target_obs[6])
        targs = torch.max(targ_out, dim=-1)[0][:,None]
        rewards = torch.Tensor(rewards)[:,None].to(self.device)
        dones = torch.Tensor(dones)[:,None].to(self.device)

        targs = rewards + self.args['disc_rate'] * (1-dones) * targs
        self.target_vals.append(targs)

        self.target_hidden_edge = t_e_hid
        self.target_hidden_node = t_n_hid
        self.target_hidden_u = list(zip([hid[None, None, :] for hid in t_u_hid[0][0]],
                                        [hid[None, None, :] for hid in t_u_hid[1][0]]))

    def prep_obs(self, obses, prev_hidden_e, prev_hidden_n, prev_hidden_u):
        # All the list used for creating batches
        new_graphs = []
        edge_feature_list = []
        node_feature_list = []
        u_feature_list = []
        prep_hidden_e_list = []
        prep_hidden_n_list = []

        for idx, obs in enumerate(obses):
            if prev_hidden_e[idx] is None:
                prev_hidden_e[idx] = (torch.zeros([1, len(obs[2]) * (len(obs[2]) - 1), 10]).to(self.device),
                                         torch.zeros([1, len(obs[2]) * (len(obs[2]) - 1), 10]).to(self.device))
            if prev_hidden_n[idx] is None:
                prev_hidden_n[idx] = (torch.zeros([1, len(obs[2]), 10]).to(self.device),
                                      torch.zeros([1, len(obs[2]), 10]).to(self.device))
            if prev_hidden_u[idx] is None:
                prev_hidden_u[idx] = (torch.zeros([1, 1, 10]).to(self.device),
                                      torch.zeros([1, 1, 10]).to(self.device))

            new_graph, edge_filters = self.create_input_graph(obs[2], obs[3], idx)
            new_graphs.append(new_graph)

            # Calculate number of added nodes and new number of nodes
            added_n, new_node_num = obs[3], new_graph.nodes().shape[0]
            # Calculate number of nodes after filtering
            after_delete_node = new_node_num - added_n
            # Calculate the added number of edges
            added_e = (new_node_num * (new_node_num - 1)) - (after_delete_node * (after_delete_node - 1))

            # Prepare hiddens

            # Empty features for edge
            edge_features = torch.Tensor(size=[new_node_num * (new_node_num - 1), 0]).to(self.device)
            edge_feature_list.append(edge_features)
            # Features for nodes
            node_features = torch.Tensor(obs[0]).to(self.device)
            node_feature_list.append(node_features)
            # Use image as features for graph
            u_features = torch.Tensor(obs[1]).permute(2, 0, 1)[None, :, :, :].to(self.device)
            u_feature_list.append(u_features)

            preprocessed_hidden_e = self.prep_hidden(prev_hidden_e[idx], [edge_filters], [added_e])
            prep_hidden_e_list.append(preprocessed_hidden_e)
            preprocessed_hidden_n = self.prep_hidden(prev_hidden_n[idx], [torch.Tensor(obs[2]).to(self.device)],
                                                     [added_n])
            prep_hidden_n_list.append(preprocessed_hidden_n)

        e_feat, n_feat, u_feat = torch.cat(edge_feature_list, dim=0), \
                                 torch.cat(node_feature_list, dim=0), \
                                 torch.cat(u_feature_list, dim=0)

        hid_1_e, hid_2_e = zip(*prep_hidden_e_list)
        hid_e = (torch.cat(hid_1_e, dim=1), torch.cat(hid_2_e, dim=1))

        hid_1_n, hid_2_n = zip(*prep_hidden_n_list)
        hid_n = (torch.cat(hid_1_n, dim=1), torch.cat(hid_2_n, dim=1))

        hid_1_u, hid_2_u = zip(*self.hidden_u)
        hid_u = (torch.cat(hid_1_u, dim=1), torch.cat(hid_2_u, dim=1))

        return new_graphs, e_feat, n_feat, u_feat,\
               hid_e, hid_n, hid_u



    def reset(self, obs):
        self.hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.hidden_node = [None for _ in range(self.args['num_envs'])]
        self.hidden_u = [None for _ in range(self.args['num_envs'])]
        self.graph = [None for _ in range(self.args['num_envs'])]
        self.next_graph = [None for _ in range(self.args['num_envs'])]
        self.obs = None
        self.next_obs = None

        self.prev_hidden = None
        self.curr_hidden = None

        self.predicted_vals = []
        self.target_vals = []

        self.target_hidden_edge = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_node = [None for _ in range(self.args['num_envs'])]
        self.target_hidden_u = [None for _ in range(self.args['num_envs'])]

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def create_input_graph(self, node_filters, num_added_nodes, idx=0):
        device = torch.device('cpu') if self.device == "cpu" else torch.device('cuda:0')
        new_graph = dgl.DGLGraph()
        new_graph.add_nodes(len(node_filters))
        src, dest = tuple(zip(*[(i, j) for i in range(len(node_filters)) for j in range(len(node_filters)) if i != j]))
        src_past, dest_past = tuple(zip(*[(i, j) for i in node_filters for j in node_filters if
                                          i != j]))
        new_graph.add_edges(src, dest)

        # Compute edge filters based on added and removed nodes
        if self.graph[idx] is None:
            edge_filters = new_graph.edge_ids(src_past, dest_past).long().to(self.device)
        else:
            edge_filters = self.graph[idx].edge_ids(src_past, dest_past).long().to(self.device)

        node_filters = torch.Tensor(node_filters).long().to(self.device)
        if num_added_nodes != 0:
            added = num_added_nodes
            new_graph.add_nodes(num_added_nodes)
            new_idx = len(node_filters)
            src, dest = tuple(zip(*[(i, j) for i in range(new_idx + added) for j in range(new_idx + added)
                                    if (i in range(new_idx, new_idx + added) or j in range(new_idx, new_idx + added))
                                    and i != j]))

            new_graph.add_edges(src, dest)
        new_graph.to(device)

        return new_graph, edge_filters

    def prep_hidden(self,hidden, remain_mask, added_num):

        hid_filtered = [(hidden[0].gather(1, remain_mask[0].long()[None, :, None].repeat(1, 1, self.dim_lstm_out)),
                           hidden[1].gather(1, remain_mask[0].long()[None, :, None].repeat(1, 1, self.dim_lstm_out)))]

        hid_added = [(torch.cat([k[0], torch.zeros([1, added, self.dim_lstm_out]).to(self.device)], dim=1),
                        torch.cat([k[1], torch.zeros([1, added, self.dim_lstm_out]).to(self.device)], dim=1))
                       for k, added in zip(hid_filtered, added_num)]

        e_hid_1, e_hid_2 = zip(*hid_added)
        hidden_processed = (torch.cat(e_hid_1, dim=1), torch.cat(e_hid_2, dim=1))

        return hidden_processed

    def update(self):
        self.optimizer.zero_grad()
        pred_tensor = torch.cat(self.predicted_vals, dim = 0)
        target_tensor = torch.cat(self.target_vals, dim = 0)
        loss = self.loss_module(pred_tensor, target_tensor.detach())
        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net, self.args['tau'])
        self.predicted_vals = []
        self.target_vals = []

        self.curr_hidden = ((self.curr_hidden[0][0].detach(), self.curr_hidden[0][1].detach()),
                            (self.curr_hidden[1][0].detach(), self.curr_hidden[1][1].detach()),
                            (self.curr_hidden[2][0].detach(), self.curr_hidden[2][1].detach()))
