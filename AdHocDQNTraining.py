
import copy
from random import sample
import pickle as pkl
import argparse
import torch
from Agent import *
#import pygame
import ray
from QNetwork import *

# import ray
#import multiprocessing as mp
import timeit
#import matplotlib.pyplot as plt


class Generator(object):
    def __init__(self, size, deathLimit=4, birthLimit=3):
        self.x_size = size[0]
        self.y_size = size[1]
        self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
        self.probStartAlive = 0.82;
        self.deathLimit = deathLimit
        self.birthLimit = birthLimit
        self.copy = None

    def initialiseMap(self):
        for x in range(self.x_size):
            for y in range(self.y_size):
                if random.random() < self.probStartAlive:
                    self.booleanMap[y][x] = True

    def doSimulationStep(self):

        newMap = [[False] * k for k in [self.x_size] * self.y_size]
        for x in range(self.x_size):
            for y in range(self.y_size):
                alive = self.countAliveNeighbours(x, y)
                if self.booleanMap[y][x]:
                    if alive < self.deathLimit:
                        newMap[y][x] = False
                    else:
                        newMap[y][x] = True
                else:
                    if alive > self.birthLimit:
                        newMap[y][x] = True
                    else:
                        newMap[y][x] = False
        self.booleanMap = newMap

    def countAliveNeighbours(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbour_x = x + i
                neighbour_y = y + j
                if not ((i == 0) and (j == 0)):
                    if neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= self.x_size or neighbour_y >= self.y_size:
                        count = count + 1
                    elif self.booleanMap[neighbour_y][neighbour_x]:
                        count = count + 1
        return count

    def simulate(self, numSteps):
        done = False
        while not done:
            self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
            self.initialiseMap()
            for kk in range(numSteps):
                self.doSimulationStep()

            if self.doFloodfill(self.booleanMap):
                done = True

    def doFloodfill(self, newMap):
        self.copy = copy.deepcopy(newMap)
        foundX, foundY = -1, -1
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                if not self.copy[i][j]:
                    foundX = i
                    foundY = j
                    flag = True
                    break
            if flag:
                break
        self.floodfill(foundX, foundY)
        done = True
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                # print(self.copy[i][j])
                if not self.copy[i][j]:
                    done = False
                    flag = True
                    break
            if flag:
                break
        return done

    def floodfill(self, x, y):
        queue = []
        queue.append((x, y))
        while len(queue) != 0:
            a = queue[0][0]
            b = queue[0][1]

            del queue[0]
            if not self.copy[a][b]:
                self.copy[a][b] = True

            if (not a + 1 >= len(self.copy)) and (not self.copy[a + 1][b]):
                queue.append((a + 1, b))
                self.copy[a + 1][b] = True
            if (not (a - 1 < 0)) and (not self.copy[a - 1][b]):
                queue.append((a - 1, b))
                self.copy[a - 1][b] = True
            if (not b + 1 >= len(self.copy[0])) and (not self.copy[a][b + 1]):
                queue.append((a, b + 1))
                self.copy[a][b + 1] = True
            if (not b - 1 < 0) and (not self.copy[a][b - 1]):
                queue.append((a, b - 1))
                self.copy[a][b - 1] = True


class OpenScheduler(object):
    def __init__(self, num_agents, add_rate, remove_rate):
        self.available_agents = num_agents

        # Use geometric distribution to sample remove or del
        self.geometric_add_rate = add_rate
        self.geometric_remove_rate = remove_rate

    def add_agents(self, agent_nums):
        new_agent = []
        new_obs_type = []
        for _ in range(agent_nums):
            agent_class = self.agent_type_sampler()
            agent = agent_class(self.available_agents + 1)
            new_agent.append(agent)
            new_obs_type.append(agent.obs_type)
            self.available_agents += 1
        return new_agent, new_obs_type

    def del_agents(self, agent_idxs):
        agent_idxs_sorted = agent_idxs.copy()
        agent_idxs_sorted.sort(reverse=True)

        self.available_agents -= len(agent_idxs_sorted)
        return agent_idxs_sorted

    def agent_type_sampler(self):
        agent_id_type = random.randint(0, 3)
        all_agent_types = [RandomAgent, GreedyPredatorAgent, GreedyProbabilisticAgent,
                           TeammateAwarePredator, DistilledCoopAStarAgent, MADDPGAgent]
        return all_agent_types[agent_id_type]

    def agent_removal_sampler(self):
        removed_amount = np.random.choice(2, 1, p=[0.7, 0.3])[0] + 1
        removed_indices = random.sample(list(range(self.available_agents)),
                                        k=min(self.available_agents - 1, removed_amount))
        removed_indices = [k + 1 for k in removed_indices]
        return removed_indices

    def open_process(self):
        remove = False
        if random.random() < self.geometric_remove_rate:
            remove = True

        add = False
        if random.random() < self.geometric_add_rate:
            add = True

        deleted_idxs = []
        if remove:
            removed_indices = self.agent_removal_sampler()
            deleted_idxs = self.del_agents(removed_indices)

        new_agents, new_obs_type = [], []
        if add:
            agent_nums = np.random.choice(2, 1, p=[0.7, 0.3])[0] + 1
            new_agents, new_obs_type = self.add_agents(agent_nums)
        return deleted_idxs, new_agents, new_obs_type


class Wolfpack(object):
    def __init__(self, grid_height, grid_width, agent_list, sight_sideways=8,
                 sight_radius=8, num_players=5, max_food_num=2, food_freeze_rate=0,
                 max_time_steps=200, coop_radius=4, groupMultiplier=2, scheduler=None):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.scheduler = scheduler
        self.agent_list = agent_list

        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.pads = max(self.sight_sideways, self.sight_radius)

        self.RGB_padded_grid = [[[0, 0, 255] for b in range(2 * self.pads + self.grid_width)]
                                for a in range(2 * self.pads + self.grid_height)]

        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
        self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]
        self.num_players = num_players
        self.max_food_num = max_food_num
        self.max_time_steps = max_time_steps
        self.food_freeze_rate = food_freeze_rate

        app = Generator((self.grid_width, self.grid_height), 7, 8)
        app.initialiseMap()
        app.simulate(2)
        self.levelMap = app.booleanMap
        #self.visualizer = Visualizer(self.grid, self.grid_height, self.grid_width)

        self.obstacleCoord = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if i]
        self.possibleCoordinates = None
        self.player_positions = None
        self.player_orientation = None
        self.food_positions = None
        self.food_alive_statuses = None
        self.food_frozen_time = None
        self.food_orientation = None
        self.player_points = None
        self.food_points = None
        self.a_to_idx = None
        self.idx_to_a = None

        self.player_obs_type = []
        self.food_obs_type = []

        # Find out about this
        # Find out about this
        self.remaining_timesteps = max_time_steps

        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.coopRadius = coop_radius
        self.groupMultiplier = groupMultiplier

        if not self.scheduler is None:
            self.prev_added = 0
            self.masking = []


    def save_map(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.levelMap, f)

    def load_map(self, filename):
        with open(filename, 'rb') as f:
            self.levelMap = pkl.load(f)

    def reset(self, agent_list):
        self.agent_list = agent_list
        self.player_obs_type = [a.obs_type for a in self.agent_list]
        self.num_players = len(self.agent_list)
        if not self.scheduler is None:
            self.scheduler.available_agents = self.num_players - 1
            self.prev_added = 0
            self.masking = [i for i in range(self.num_players)]
        self.RGB_padded_grid = [[[0, 0, 255] for b in range(2 * self.pads + self.grid_width)]
                                for a in range(2 * self.pads + self.grid_height)]
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]
        self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]
        self.possibleCoordinates = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if
                                    not i]

        player_loc_idx = sample(range(len(self.possibleCoordinates)), self.num_players)
        self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]
        self.player_orientation = [0 for a in range(self.num_players)]
        self.player_points = [0 for a in range(self.num_players)]

        coordinates_no_player = [a for a in self.possibleCoordinates if a not in self.player_positions]
        food_loc_idx = sample(range(len(coordinates_no_player)), self.max_food_num)

        self.food_positions = [coordinates_no_player[a] for a in food_loc_idx]
        self.food_alive_statuses = [True for a in range(self.max_food_num)]
        self.food_frozen_time = [0 for a in range(self.max_food_num)]
        self.food_points = [0 for a in range(self.max_food_num)]

        self.food_orientation = [0 for a in range(self.max_food_num)]

        for coord in self.possibleCoordinates:
            self.grid[coord[0]][coord[1]] = 1
            self.RGB_grid[coord[0]][coord[1]] = [0, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [0, 0, 0]
        for coord in self.player_positions:
            self.grid[coord[0]][coord[1]] = 2
            self.RGB_grid[coord[0]][coord[1]] = [255, 255, 255]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 255, 255]
        for coord in self.food_positions:
            self.grid[coord[0]][coord[1]] = 3
            self.RGB_grid[coord[0]][coord[1]] = [255, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 0, 0]

        self.remaining_timesteps = self.max_time_steps

        return [self.observation_computation(obs_type, agent_id=id) for id, obs_type in
                enumerate(self.player_obs_type)], [self.observation_computation(obs_type,
                agent_type="food", agent_id=id) for id, obs_type in enumerate(self.food_obs_type)]

    def revive(self):
        # find possible locations to revive dead player
        coordinates_no_player = [a for a in self.possibleCoordinates if
                                 a not in self.player_positions and a not in self.food_positions]
        revived_idxes = []
        for idx, food in enumerate(self.food_positions):
            if self.food_frozen_time[idx] <= 0 and not self.food_alive_statuses[idx]:
                revived_idxes.append(idx)

        if len(revived_idxes) > 0:
            idxes = []
            for k in range(len(revived_idxes)):
                idx = sample(range(len(coordinates_no_player)), 1)[0]
                while idx in idxes:
                    idx = sample(range(len(coordinates_no_player)), 1)[0]
                idxes.append(idx)
            coords = [coordinates_no_player[idx] for idx in idxes]

            coord_idx = 0
            for idx in revived_idxes:
                self.food_alive_statuses[idx] = True
                self.food_positions[idx] = coords[coord_idx]
                coord_idx += 1

    def update_status(self):
        for idx in range(len(self.food_alive_statuses)):
            if not self.food_alive_statuses[idx]:
                self.food_frozen_time[idx] -= 1

    def calculate_new_position(self, collectiveAct, prev_player_position, prev_player_orientation):
        zipped_data = list(zip(collectiveAct, prev_player_position, prev_player_orientation))
        result = [self.calculate_indiv_position(a, (b, c), d) for (a, (b, c), d) in zipped_data]
        return result

    def calculate_indiv_position(self, action, pair, orientation):
        x = pair[0]
        y = pair[1]
        next_x = x
        next_y = y

        # go forward
        if action == 0:
            # Facing upwards
            if orientation == 0:
                next_x -= 1
            # Facing right
            elif orientation == 1:
                next_y += 1
            # Facing downwards
            elif orientation == 2:
                next_x += 1
            else:
                next_y -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step right
        elif action == 1:
            # Facing upwards
            if orientation == 0:
                next_y += 1
            # Facing right
            elif orientation == 1:
                next_x += 1
            # Facing downwards
            elif orientation == 2:
                next_y -= 1
            else:
                next_x -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step back
        elif action == 2:
            # Facing upwards
            if orientation == 0:
                next_x += 1
            # Facing right
            elif orientation == 1:
                next_y -= 1
            # Facing downwards
            elif orientation == 2:
                next_x -= 1
            else:
                next_y += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step left
        elif action == 3:
            # Facing upwards
            if orientation == 0:
                next_y -= 1
            # Facing right
            elif orientation == 1:
                next_x -= 1
            # Facing downwards
            elif orientation == 2:
                next_y += 1
            else:
                next_x += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # stay still
        elif action == 4:
            return (x, y, orientation)
        # rotate left
        elif action == 5:
            new_orientation = 0
            if orientation == 0:
                new_orientation = 3
            elif orientation == 1:
                new_orientation = 0
            elif orientation == 2:
                new_orientation = 1
            else:
                new_orientation = 2

            return (x, y, new_orientation)
        # rotate right
        else:
            new_orientation = 0
            if orientation == 0:
                new_orientation = 1
            elif orientation == 1:
                new_orientation = 2
            elif orientation == 2:
                new_orientation = 3
            else:
                new_orientation = 0

            return (x, y, new_orientation)

    def update_food_status(self):
        self.food_points = [0 for a in range(self.max_food_num)]

        enumFood = list(enumerate(self.food_positions))
        food_locations = [(food[0], food[1]) for idx, food in enumFood if self.food_alive_statuses[idx]]
        food_id = [idx for idx, food in enumFood if self.food_alive_statuses[idx]]

        player_locations = self.player_positions
        set_of_food_location = set(food_locations)

        self.player_points = [0 for a in range(self.num_players)]
        self.food_points = [0 for a in range(self.max_food_num)]

        for player_loc in player_locations:
            if player_loc in set_of_food_location:
                center = player_loc
                enumerated = enumerate(player_locations)
                close = [x for (x, (a, b)) in enumerated if abs(a - center[0]) + abs(b - center[1]) <= self.coopRadius]
                for x in close:
                    if len(close) < 2:
                        self.player_points[x] += len(close)
                    else:
                        self.player_points[x] += self.groupMultiplier * len(close)
                food_index = food_locations.index(center)
                self.food_points[food_id[food_index]] += -1
                self.food_alive_statuses[food_id[food_index]] = False
                self.food_frozen_time[food_id[food_index]] = self.food_freeze_rate

        for idx, food in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = 3
                self.RGB_grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = [255, 0, 0]
                self.RGB_padded_grid[self.food_positions[idx][0] + self.pads][self.food_positions[idx][1] + self.pads] \
                    = [255, 0, 0]

    def update_state(self, hunter_collective_action, food_collective_action):
        self.remaining_timesteps -= 1
        self.update_status()
        self.revive()

        prev_player_position = self.player_positions
        prev_player_orientation = self.player_orientation
        prev_food_position = self.food_positions
        prev_food_orientation = self.food_orientation

        # Calculate new player positions
        # -here
        update_results = self.calculate_new_position(hunter_collective_action, prev_player_position,
                                                     prev_player_orientation)
        post_player_position = [(a, b) for (a, b, c) in update_results]
        post_player_orientation = [c for (a, b, c) in update_results]
        self.player_orientation = post_player_orientation

        # Calculate player intersection
        a, seen, result = post_player_position, set(), {}
        for idx, item in enumerate(a):
            if item not in seen:
                result[item] = [idx]
                seen.add(item)
            else:
                result[item].append(idx)

        groupings = list(result.values())
        doubles = [t for t in groupings if len(t) > 1]
        while len(doubles) > 0:
            res = set([item for sublist in doubles for item in sublist])
            for ii in range(len(self.player_positions)):
                if ii in res:
                    post_player_position[ii] = prev_player_position[ii]

            a, seen, result = post_player_position, set(), {}
            for idx, item in enumerate(a):
                if item not in seen:
                    result[item] = [idx]
                    seen.add(item)
                else:
                    result[item].append(idx)

            groupings = list(result.values())
            doubles = [t for t in groupings if len(t) > 1]

        for a in self.player_positions:
            self.grid[a[0]][a[1]] = 1
            self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.player_positions = post_player_position

        # Calculate new food locations
        update_results = self.calculate_new_position(food_collective_action, prev_food_position, prev_food_orientation)
        post_food_position = [(a, b) for (a, b, c) in update_results]
        post_food_orientation = [c for (a, b, c) in update_results]
        self.food_orientation = post_food_orientation

        # # Calculate food intersection
        # a, seen, result = post_food_position, set(), {}
        # for idx, item in enumerate(a):
        #     if self.food_alive_statuses:
        #         if item not in seen:
        #             result[item] = [idx]
        #             seen.add(item)
        #         else:
        #             result[item].append(idx)
        #
        # groupings = list(result.values())
        # doubles = [t for t in groupings if len(t) > 1]
        # res = set([item for sublist in doubles for item in sublist])
        # for ii in range(len(self.food_positions)):
        #     if ii not in res:
        #         if self.grid[prev_food_position[ii][0]][prev_food_position[ii][1]] != 2:
        #             self.grid[prev_food_position[ii][0]][prev_food_position[ii][1]] = 1
        #         self.food_positions[ii] = post_food_position[ii]
        #     else:
        #         self.food_positions[ii] = prev_food_position[ii]

        a, seen, result = post_food_position, set(), {}
        for idx, item in enumerate(a):
            if self.food_alive_statuses[idx]:
                if item not in seen:
                    result[item] = [idx]
                    seen.add(item)
                else:
                    result[item].append(idx)

        groupings = list(result.values())
        doubles = [t for t in groupings if len(t) > 1]
        while len(doubles) > 0:

            res = set([item for sublist in doubles for item in sublist])
            for ii in range(len(post_food_position)):
                if ii in res:
                    post_food_position[ii] = prev_food_position[ii]

            a, seen, result = post_food_position, set(), {}
            for idx, item in enumerate(a):
                if self.food_alive_statuses[idx]:
                    if item not in seen:
                        result[item] = [idx]
                        seen.add(item)
                    else:
                        result[item].append(idx)

            groupings = list(result.values())
            doubles = [t for t in groupings if len(t) > 1]

        for a in self.food_positions:
            self.grid[a[0]][a[1]] = 1
            self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.food_positions = post_food_position

        for idx, a in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[a[0]][a[1]] = 3
                self.RGB_grid[a[0]][a[1]] = [255, 0, 0]
                self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 0, 0]
        for a in self.player_positions:
            self.grid[a[0]][a[1]] = 2
            self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
        # I'm here
        # Calculate player points and food status
        self.update_food_status()

    def observation_computation(self, obs_type, agent_type="player", agent_id=0):
        if obs_type == "comp_processed":
            return [self.player_positions, self.player_orientation,
                    self.food_positions, self.food_orientation, self.food_alive_statuses,
                    self.possibleCoordinates]
        elif obs_type == "full_rgb":
            if agent_type == "player":
                orientation = self.player_orientation[agent_id]
                position = self.player_positions[agent_id]
            else:
                orientation = self.food_orientation[agent_id]
                position = self.player_positions[agent_id]

            original_state = np.asarray(self.RGB_grid)

            pos_list = list(position)
            orientation_list = [0] * 4
            orientation_list[orientation] = 1
            pos_list.extend(orientation_list)
            return (original_state, pos_list)
        elif obs_type == "full_rgb_graph":
            orientation = self.player_orientation[agent_id]
            position = self.player_positions[agent_id]

            original_state = np.asarray(self.RGB_grid)

            def orientation_to_one_hot(orientation):
                orientation_list = [0] * 4
                orientation_list[orientation] = 1
                return orientation_list

            own_pos = list(position)
            own_pos.extend(orientation_to_one_hot(orientation))
            position_list = [own_pos]
            enemy_pos_list = self.food_positions

            for idx in range(len(self.player_orientation)):
                if idx != agent_id:
                    other_pos = list(self.player_positions[idx])
                    other_pos.extend(orientation_to_one_hot(self.player_orientation[idx]))
                    position_list.append(other_pos)

            # pos_list.extend(orientation_list)
            return (original_state, position_list, enemy_pos_list)

        elif obs_type == "partial_obs":
            if agent_type == "player":
                orientation = self.player_orientation[agent_id]
                pos_0, pos_1 = self.player_positions[agent_id][0], self.player_positions[agent_id][1]
            else:
                orientation = self.food_orientation[agent_id]
                pos_0, pos_1 = self.food_positions[agent_id][0], self.food_positions[agent_id][1]

            pos_0 = pos_0 + self.pads
            pos_1 = pos_1 + self.pads
            obs_grid = np.asarray(self.RGB_padded_grid)

            if orientation == 0:
                partial_ob = obs_grid[pos_0 - self.sight_radius:pos_0 + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]


            elif orientation == 1:
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1:pos_1 + self.sight_radius + 1]

                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = partial_ob[::-1]

            elif orientation == 2:
                partial_ob = obs_grid[pos_0:pos_0 + self.sight_radius + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
                partial_ob = np.fliplr(partial_ob)
                partial_ob = partial_ob[::-1]

            elif orientation == 3:
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1 - self.sight_radius:pos_1 + 1]
                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = np.fliplr(partial_ob)

            return partial_ob
        elif obs_type == "adhoc_obs":
            orientation = self.player_orientation
            positions = self.player_positions

            pos_list = [list(pos_data) for pos_data in positions]
            for pos, ord in zip(pos_list, orientation):
                orients = [0] * 4
                orients[ord] = 1
                pos.extend(orients)

            prob_state = np.asarray(self.RGB_grid)

            return pos_list, prob_state, self.masking, self.prev_added

        # elif obs_type == "centralized_decentralized":

    def add_agent(self, new_agent, new_types):
        def_orientation = 0
        available_pos = list(set(self.possibleCoordinates) - set(self.player_positions) -
                             set(self.food_positions))
        pos_idxes = random.sample(list(range(len(available_pos))), k=len(new_agent))
        added_pos = [available_pos[a] for a in pos_idxes]
        orientation = [def_orientation for _ in range(len(added_pos))]

        self.player_orientation.extend(orientation)
        self.player_positions.extend(added_pos)
        self.player_obs_type.extend(new_types)
        self.agent_list.extend(new_agent)
        for a in added_pos:
            self.grid[a[0]][a[1]] = 2
            self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
            self.num_players += 1

    def del_agent(self, agent_id):
        for idx in agent_id:
            a = self.player_positions[idx]
            self.grid[a[0]][a[1]] = 1
            self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]

            self.player_positions.pop(idx)
            self.agent_list.pop(idx)
            self.player_orientation.pop(idx)
            self.player_obs_type.pop(idx)
            self.num_players -= 1

    def step(self, hunter_collective_action, food_collective_action):
        self.update_state(hunter_collective_action, food_collective_action)
        if not self.scheduler is None:
            deleted, agents, new_types = self.scheduler.open_process()
            self.masking = [idx for idx in range(len(self.agent_list)) if not idx in deleted]
            self.del_agent(deleted)
            self.add_agent(agents, new_types)
            for idx, agent in enumerate(self.agent_list):
                agent.agent_id = idx
            self.prev_added = len(agents)

        player_returns = ([self.observation_computation(obs_type, agent_id=id)
                           for id, obs_type in enumerate(self.player_obs_type)],
                          self.player_points, [self.remaining_timesteps == 0 for
                                               a in range(len(self.player_points))])

        food_returns = ([self.observation_computation(obs_type, agent_type="food", agent_id=id)
                         for id, obs_type in enumerate(self.food_obs_type)],
                        self.food_points, [self.remaining_timesteps == 0
                                           for a in range(self.max_food_num)])

        #self.visualizer.grid = self.grid

        return player_returns, food_returns
    def set_agent_obs_type(self, list_obs):
        self.player_obs_type = list_obs

    def set_food_obs_type(self, list_obs):
        self.food_obs_type = list_obs

    def sample_init_players(self):
        num_sampled = random.randint(1, 4)
        all_agent_types = [RandomAgent, GreedyPredatorAgent]
        agent_inits = [all_agent_types[random.randint(0, len(all_agent_types) - 1)](idx + 1)
                       for idx in range(num_sampled)]
        return agent_inits

    #def render(self):
    #    self.visualizer.render()

#@ray.remote
#def get_food(food, obs):
#    return food.act(obs)

class AdHocWolfpack(Wolfpack):
    def __init__(self, grid_height, grid_width, agent, args= None, sight_sideways=8,
                 sight_radius=8, num_players=5, max_food_num=2, food_freeze_rate=0,
                 max_time_steps=200, coop_radius=4, groupMultiplier=2, scheduler=None):
        self.agent = agent
        agent_list = [agent]
        Wolfpack.__init__(self, grid_height, grid_width, agent_list, sight_sideways,
                 sight_radius, num_players, max_food_num, food_freeze_rate,
                 max_time_steps, coop_radius, groupMultiplier, scheduler)

        self.args = args
        self.food_list = None
        self.add_foods()
        self.food_obs_type = [a.obs_type for a in self.food_list]
        self.food_obs = None

        self.other_player_obs = None

    def step(self, hunter_action):
        food_collective_action = [food.act(ob) for food, ob in zip(self.food_list, self.food_obs)]
        hunter_collective_action = [hunter_action]
        others_actions = [other.act(ob) for other, ob in zip(self.other_players, self.other_player_obs)]
        hunter_collective_action.extend(others_actions)
        player_returns, food_returns = Wolfpack.step(self, hunter_collective_action,
                                                                       food_collective_action)
        self.other_players = []
        if len(self.agent_list) > 1:
            self.other_players = self.agent_list[1:]
        self.food_obs = food_returns[0]
        self.other_player_obs = player_returns[0][1:]
        return (player_returns[0][0],player_returns[1][0], player_returns[2][0])

    def reset(self):
        self.add_foods()
        self.other_players = self.sample_init_players()
        agent_list = [self.agent]
        agent_list.extend(self.other_players)
        player_obs, food_obs = Wolfpack.reset(self, agent_list)
        self.food_obs = food_obs
        self.other_player_obs = player_obs[1:]
        return player_obs[0]

    def add_foods(self):
        self.food_list = [DQNAgent(agent_id = a, args=self.args, obs_type="partial_obs")
                          for a in range(self.max_food_num)]
        for idx, agent in enumerate(self.food_list):
            agent.load_parameters("assets/dqn_prey_parameters/exp0.0001param_10_agent_"+str(idx))

class Visualizer(object):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    WIDTH = 20
    HEIGHT = 20

    MARGIN = 0

    # Create a 2 dimensional array. A two dimensional
    # array is simply a list of lists.

    def __init__(self, grid, grid_height=20, grid_width=20):
        self.grid_height, self.grid_width = grid_height, grid_width
        self.grid = grid
        self.WINDOW_SIZE = [self.grid_height * self.HEIGHT, self.grid_width * self.WIDTH]

        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Wolfpack")
        self.clock = pygame.time.Clock()

    def render(self):
        done = False
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)

        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLUE
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                elif self.grid[row][column] == 3:
                    color = self.RED
                pygame.draw.rect(self.screen,
                                     color,
                                     [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                      (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                      self.WIDTH,
                                      self.HEIGHT])

        self.clock.tick(60)
        pygame.display.flip()

        if done:
            pygame.quit()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64 ,help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--disc_rate', type=float, default=0.99, help='dicount_rate')
parser.add_argument('--exp_replay_capacity', type=int, default=1e5, help='experience replay capacity')
parser.add_argument('--num_predators', type=int, default=2, help='number of predators')
parser.add_argument('--num_food', type=int, default=2, help='number of preys')
parser.add_argument('--max_bptt_length', type=int, default=20, help="length of state sequence")
parser.add_argument('--num_episodes', type=int, default=2500, help="Number of episodes for training")
parser.add_argument('--update_frequency', type=int, default=32, help="Timesteps between updates")
parser.add_argument('--episode_length', type=int, default=200, help="Number of timesteps in episode")
parser.add_argument('--anneal_end', type=int, default=4000, help="Number of episodes until linear annealing stops")
parser.add_argument('--sampling_wait_time', type=int, default=100, help="timesteps until begin updating")
parser.add_argument('--saving_frequency', type=int,default=100,help="saving frequency")
parser.add_argument('--obs_height', type=int,default=9,help="observation_height")
parser.add_argument('--obs_width', type=int,default=17,help="observation_width")
parser.add_argument('--obs_type', type=str,default="partial_obs",help="observation type")
parser.add_argument('--with_gpu', type=bool,default=False,help="with gpu")
parser.add_argument('--add_rate', type=float,default=0.05,help="agent added rate")
parser.add_argument('--del_rate', type=float,default=0.05,help="agent deletion rate")
parser.add_argument('--num_envs', type=int,default=16, help="Number of environments")
parser.add_argument('--tau', type=float,default=0.001, help="tau")
parser.add_argument('--max_seq_length', type=int, default=10, help="length of training sequence")
parser.add_argument('--maddpg_max_seq_length', type=int, default=10, help="length of state sequence")


args = parser.parse_args()

def create_parallel_env(args, player, num_envs):
    schedulers = [OpenScheduler(4, args['add_rate'], args['del_rate']) for _ in range(num_envs)]
    envs = [AdHocWolfpack(25, 25, agent=player, args=args,
                            num_players=args['num_predators'], max_food_num=args['num_food'],
                            max_time_steps=args['episode_length'], scheduler=scheduler) for scheduler
                     in schedulers]
    for env in envs:
        env.load_map('level_1.pkl')

    return envs

if __name__ == '__main__':
    add_rate = 0.05
    rem_rate = 0.05
    torch.set_num_threads(1)
    arguments = vars(args)

    player = AdHocDQNAgent(args=arguments, agent_id=0)

    num_episodes = arguments['num_episodes']
    eps_length = arguments['episode_length']

    scheduler = OpenScheduler(4, arguments['add_rate'], arguments['del_rate'])
    env = AdHocWolfpack(25, 25, agent=player, args=arguments,
                            num_players=arguments['num_predators'], max_food_num=arguments['num_food'],
                            max_time_steps=arguments['episode_length'], scheduler=scheduler)
    env.load_map('level_1.pkl')
    # Setup

    total_timesteps = 0


    for eps_index in range(num_episodes):
        start = timeit.default_timer()
        num_timesteps = 0
        env_obs = [env.reset()]
        player.reset(env_obs)
        done = False
        total_update_time = 0
        while not done:
            player_act = player.step(env_obs)
            player_obs = env.step(player_act)
            next_obs = [player_obs[0]]
            rewards = [player_obs[1]]
            dones = [player_obs[2]]

            player.set_next_state(env_obs, player_act, rewards[0], dones[0], next_obs)

            env_obs = next_obs
            done = dones[0]
            num_timesteps += 1
            total_timesteps += 1
            if total_timesteps % arguments['update_frequency'] == 0:
                player.update()

        end = timeit.default_timer()
        print("Eps Done!!! Took these seconds : ", str(end-start), " with total update time : ", str(total_update_time))
        player.set_epsilon(max(1.0-((eps_index+1.0)/1250.0), 0.05))
        if (eps_index+1)%arguments['saving_frequency'] == 0:
            player.save_parameters("parameters/params_"+str((eps_index+1)//arguments['saving_frequency']))



# if __name__ == "__main__":
#     num_players = 8
#     num_food = 2
#
#     add_rate = 0.05
#     rem_rate = 0.05
#
#     arguments = vars(args)
#
#     num_episodes = arguments['num_episodes']
#     eps_length = arguments['episode_length']
#
#     agent_list = [MADDPGAgent(idx, args=arguments) for idx in range(num_players)]
#     #agent_list = [TeammateAwarePredator(idx) for idx in range(num_players)]
#     for agent in agent_list:
#         #agent.load_params("distilled_net.pkl")
#         agent.load_params()
#     food_list = [DQNAgent(agent_id=a, args=arguments, obs_type="partial_obs")
#                       for a in range(arguments['num_food'])]
#     for agent in food_list:
#         agent.load_parameters("parameters/zipped_dqn_preys/exp0.0001param_10_agent_1")
#     food_obs_type = [a.obs_type for a in food_list]
#
#     env =  Wolfpack(25, 25, agent_list, num_players=num_players, max_food_num=num_food)
#     env.load_map('level_1.pkl')
#     env.set_food_obs_type(food_obs_type)
#     exp = []
#
#     for eps_idx in range(arguments['num_episodes']):
#         player_obs, food_obs = env.reset(agent_list)
#         env.render()
#         dones = False
#         num_steps = 0
#         while not dones:
#             food_collective_action = [food.act(ob) for food, ob in zip(food_list, food_obs)]
#             hunter_collective_action = [hunter.act(ob) for hunter, ob in zip(agent_list,
#                                                                            player_obs)]
#             #print(hunter_collective_action)
#             #print(env.player_positions)
#             data = {"img" : env.RGB_grid, "agent_pos" : env.player_positions,
#                     "agent_orientation" : env.player_orientation, "action" : hunter_collective_action,
#                     "food_positions": env.food_positions, "food_orientations" : env.food_orientation}
#             exp.append(data)
#             player_returns, food_returns = env.step(hunter_collective_action,
#                                                   food_collective_action)
#             env.render()
#             num_steps += 1
#             food_obs = food_returns[0]
#             player_obs = player_returns[0]
#             dones = player_returns[2][0]
#             #print(num_steps)
#
#     with open('episodes7.pkl', 'wb') as f:
#         pickle.dump(exp, f)
