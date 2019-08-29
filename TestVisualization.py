import argparse
from AdHocDQNTraining import AdHocWolfpack, OpenScheduler
from Agent import *
from QNetwork import *

import timeit

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default= 32 ,help='batch size')
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
parser.add_argument('--num_envs', type=int,default=1, help="Number of environments")
parser.add_argument('--tau', type=float,default=0.001, help="tau")
parser.add_argument('--max_seq_length', type=int, default=5, help="length of training sequence")
parser.add_argument('--maddpg_max_seq_length', type=int, default=10, help="length of state sequence")
parser.add_argument('--filename', type=str, default="params_collection/parameters/params_25")


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

    player = AdHocShortBPTTAgent(args=arguments, agent_id=0)
    player.set_epsilon(0.0)
    player.load_parameters(arguments['filename'])

    num_episodes = arguments['num_episodes']
    eps_length = arguments['episode_length']

    scheduler = OpenScheduler(4, arguments['add_rate'], arguments['del_rate'])
    env = AdHocWolfpack(25, 25, agent=player, args=arguments,
                            num_players=arguments['num_predators'], max_food_num=arguments['num_food'],
                            max_time_steps=arguments['episode_length'], scheduler=scheduler, with_vis=True)
    env.load_map('level_1.pkl')
    # Setup

    total_timesteps = 0


    for eps_index in range(num_episodes):
        start = timeit.default_timer()
        num_timesteps = 0
        env_obs = [env.reset()]
        player.reset(env_obs)
        env.render()
        done = False
        total_update_time = 0
        while not done:
            player_act = player.step(env_obs)
            print(player_act)
            player_obs = env.step(player_act[0])
            env.render()
            next_obs = [player_obs[0]]
            rewards = [player_obs[1]]
            dones = [player_obs[2]]

            player.set_next_state(next_obs, rewards, dones)

            env_obs = next_obs
            done = dones[0]
            num_timesteps += 1
            total_timesteps += 1

        end = timeit.default_timer()
        print("Eps Done!!! Took these seconds : ", str(end-start), " with total update time : ", str(total_update_time))
        if (eps_index+1)%arguments['saving_frequency'] == 0:
            player.save_parameters("parameters/params_"+str((eps_index+1)//arguments['saving_frequency']))
