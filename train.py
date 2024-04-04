import argparse
import numpy as np
import time
import pickle
from icecream import ic
import matplotlib.pyplot as plt
import datetime
import os

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
#import tensorflow
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow 
    tensorflow.get_logger().setLevel('INFO')
    tensorflow.autograph.set_verbosity(0)
    import logging
    tensorflow.get_logger().setLevel(logging.ERROR)
    return tensorflow

tensorflow = import_tensorflow()


import tensorflow.compat.v1.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="survey_region_maddpg", help="coverage")
    parser.add_argument("--max-episode-len", type=int, default=300, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.012, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--log-dir", type=str, default="./my_logs", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_actor(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tensorflow.variable_scope(scope, reuse=reuse):
        input_shape = (1, 10, 10, 7)
        conv1 = tensorflow.compat.v1.layers.Conv2D(filters=2, kernel_size=2, input_shape=input_shape[1:],data_format = "channels_last", padding='valid', activation=tensorflow.nn.relu)(input)
        # Second Conv3D layer
        conv2 = tensorflow.compat.v1.layers.Conv2D(filters=2, kernel_size=2,input_shape=input_shape[1:],data_format = "channels_last", padding='valid', activation=tensorflow.nn.relu)(conv1)
        # Flatten the output of the second convolutional layer
        flat = tensorflow.layers.Flatten()(conv2)

        # First dense layer
        dense1 = tensorflow.layers.Dense(units=num_units, activation=tensorflow.nn.relu)(flat)

        # Second dense layer (output layer)
        outputs = tensorflow.layers.Dense(units=num_outputs, activation=None)(dense1)
        # out = input

        return outputs

def mlp_critic(input_obs, input_act,num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tensorflow.compat.v1.variable_scope(scope, reuse=reuse):
        #input_shape = (1, 10, 10, 7)
        conv1 = tensorflow.compat.v1.layers.Conv2D(filters=3, kernel_size=8,data_format = "channels_last", padding='valid', activation=tensorflow.nn.relu)(input_obs)
        # Second Conv3D layer
        conv2 = tensorflow.compat.v1.layers.Conv2D(filters=2, kernel_size=4,data_format = "channels_last", padding='valid', activation=tensorflow.nn.relu)(conv1)
        # Flatten the output of the second convolutional layer
        flat = tensorflow.layers.Flatten()(conv2)
        ic(flat.shape)
        # Concatenate the flattened layer with another input
        for input in input_act:
            concatenated_inputs = tensorflow.concat([flat,input],axis=1)
        # First dense layer
        dense1 = tensorflow.layers.Dense(units=num_units, activation=tensorflow.nn.relu)(concatenated_inputs)

        # Second dense layer (output layer)
        outputs = tensorflow.layers.Dense(units=num_outputs, activation=None)(dense1)

        return outputs

def make_env(scenario_name, arglist, benchmark=False):
    
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios
    from multiagent.survey_environment import SurveyEnv
    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario(num_agents=2, num_obstacles=4, vision_dist=0.2, grid_resolution=10, grid_max_reward=1, reward_delta=0.001, observation_mode=obs_mode)
    # # create world
    # world = scenario.make_world()
    # create multiagent environment
    env = SurveyEnv(num_agents=2, num_obstacles=4, vision_dist=0.2, grid_resolution=32, grid_max_reward=1, reward_delta=0.001, observation_mode="image")
    env.reset()
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model_actor = mlp_actor
    model_critic = mlp_critic
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model_actor, model_critic, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model_actor, model_critic, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers




def train(arglist):

    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        env.reset()
        #env.render()
        # Create agent trainers
        #ic(env.observation_space)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        os.mkdir(arglist.save_dir,exist_ok=True)
        os.mkdir(arglist.log_dir,exist_ok=True)
        os.mkdir(arglist.plots_dir,exist_ok=True)
        os.mkdir(arglist.benchmark_dir,exist_ok=True)
        
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents

        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tensorflow.compat.v1.train.Saver()
        obs_n = env.reset()
        
        episode_step = 0
        train_step = 0
        t_start = time.time()
        #tensorboard_callback = tensorf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = arglist.log_dir + "/" + arglist.exp_name + "_" + current_time
        sess = tensorflow.compat.v1.Session()
        writer = tensorflow.compat.v1.summary.FileWriter(train_log_dir,sess.graph)
        reward_tensor = tensorflow.placeholder(tensorflow.float32, shape=(), name='reward')
        tensorflow.compat.v1.summary.scalar('reward', reward_tensor)
        agents_p_tensors = {}
        agents_q_tensors = {}
        losses_p = {}
        losses_q = {}
        names = []
        
        
        for i in range(env.n):
            agent_name = 'agent'+str(i)
            names.append(agent_name)
            agents_p_tensors[agent_name] = tensorflow.placeholder(tensorflow.float32, shape=(), name='loss_p_agent'+str(i))
            tensorflow.compat.v1.summary.scalar('loss_p_agent'+str(i), agents_p_tensors[agent_name])
            agents_q_tensors[agent_name] = tensorflow.placeholder(tensorflow.float32, shape=(), name='loss_p_agent'+str(i))
            tensorflow.compat.v1.summary.scalar('loss_q_agent'+str(i), agents_q_tensors[agent_name])
        merged = tensorflow.compat.v1.summary.merge_all()
        tensorflow.global_variables_initializer().run()
        rollout = 0
        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)  # type: (object, object, object, object)
            episode_step += 1
            


            if len(episode_rewards)%1000==0:
                plt.plot(episode_rewards)
                plt.show(block=False)

            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n
            
       
   
            for i, rew in enumerate(rew_n):

                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            
            #reward_t = tensorf.convert_to_tensor(episode_rewards[-1], dtype=tensorf.float32, dtype_hint=None, name=None) 

            #accuracy_scalar = tensorf.summary.scalar("episode_reward",episode_rewards[-1])
            #writer.add_scalar(accuracy_scalar,episode_step).eval()
            over = False
            # for i in range(0, 3):
            #     if obs_n[0][2] < -30 or obs_n[0][3] < -30 or obs_n[0][3] > 30 or obs_n[0][2] > 30:
            #         over = True

            # if over:

            #     obs_n = env.reset()
            #     episode_step = 0
            #     episode_rewards = episode_rewards[0:-1]
            #     episode_rewards.append(0)



            if done or terminal:
                print("episode_rewards: %lf train_step: %d episode number: %d"% (episode_rewards[-1], train_step,len(episode_rewards)))
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                rollout = sum(episode_rewards[:])/len(episode_rewards)
                dict_rew = {reward_tensor: rollout}
                #summary, _ = sess.run([merged_losses]+[agents_tensors[agent_name] for agent_name in names], feed_dict={agents_tensors[agent_name]:losses[agent_name] for agent_name in names})
                dict_loss_p={agents_p_tensors[agent_name]:losses_p[agent_name] for agent_name in names}
                dict_loss_q={agents_q_tensors[agent_name]:losses_q[agent_name] for agent_name in names}

                sum_1 = sess.run([merged, reward_tensor]+[agents_p_tensors[agent_name] for agent_name in names]+[agents_q_tensors[agent_name] for agent_name in names], feed_dict={**dict_rew, **dict_loss_p,**dict_loss_q})
                writer.add_summary(sum_1[0], train_step)

            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue
            # increment global step counter
            train_step += 1
            #ic(train_step)
            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    [-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = str(arglist.benchmark_dir) + str(arglist.exp_name) + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue
            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            i=0
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                agent_name = 'agent'+str(i)
                if loss is not None:
                    losses_p[agent_name] = loss[1]
                    losses_q[agent_name] = loss[0]
                else:
                    losses_p[agent_name] = 0.0
                    losses_q[agent_name] = 0.0
                i+=1

           
            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                ##
                print("steps: {}, episodes: {}, mean episode reward: {},  time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
            # else:
            #     #continue
            #     print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
            #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
            #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes. all %d steps'.format(len(episode_rewards)) %train_step)
                break



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)