from common.utilities.project import Project
import sys
from common.rl_agents.dummy_agent import DummyAgent
from common.safe_gym.safe_gym import SafeGym
from common.utilities.front_end_printer import *
from common.utilities.autoencoder_data_collector import *
import gym
import random
import math
import numpy as np
import torch
from collections import deque

import gc



def train(project, env, prop_type=''):
    all_episode_rewards = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    all_property_results = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    best_reward_of_sliding_window = -math.inf
    last_max_steps_states = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_actions = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_rewards = deque(maxlen=project.command_line_arguments['max_steps']*2)
    last_max_steps_terminals = deque(maxlen=project.command_line_arguments['max_steps']*2)
    if project.command_line_arguments['autoencoder_folder'] != "":
        m_autoencoder_data_collector = AutoencoderDataCollector(project.command_line_arguments['autoencoder_folder'])
    else:
        m_autoencoder_data_collector = None
    try:
        project.agent.load_env(env)
    except:
        pass
    try:
        for episode in range(project.command_line_arguments['num_episodes']):
            state = env.reset()
            last_max_steps_states.append(state)
            episode_reward = 0
            step_counter = 0
            while True:
                if state.__class__.__name__ == 'int':
                    state = [state]
                #print(project.command_line_arguments['deploy'])
                action = project.agent.select_action(state, project.command_line_arguments['deploy'])
                step_counter +=1
                next_state, reward, terminal, info = env.step(action)
                if next_state.__class__.__name__ == 'int':
                    next_state = [next_state]
                if project.command_line_arguments['deploy']==False:
                    project.agent.store_experience(state, action, reward, next_state, terminal)
                    project.agent.step_learn()
                elif m_autoencoder_data_collector != None and project.command_line_arguments['deploy']==True:
                    m_autoencoder_data_collector.collect_data(state, next_state)
                # Collect last max_steps states, actions, and rewards
                last_max_steps_states.append(next_state)
                last_max_steps_actions.append(action)
                last_max_steps_rewards.append(reward)
                last_max_steps_terminals.append(terminal)
                state = next_state
                episode_reward+=reward
                if terminal:
                    break
            if project.command_line_arguments['deploy']==False:
                project.agent.episodic_learn()
            if episode % project.command_line_arguments['eval_interval']==0 and project.command_line_arguments['task']=='safe_training':
                # Log reward and property result (Safe Training)
                all_episode_rewards.append(episode_reward)
                project.mlflow_bridge.log_reward(all_episode_rewards[-1], episode)
                try:
                    project.mlflow_bridge.log_reward_i(all_episode_rewards[-1], episode, project.agent.turn_value)
                except:
                    pass
                reward_of_sliding_window = np.mean(list(all_episode_rewards))
                project.mlflow_bridge.log_avg_reward(reward_of_sliding_window, episode)
                # Log Property Result
                if prop_type != 'reward':
                    mdp_reward_result, model_size = env.storm_bridge.model_checker.induced_markov_chain(project.agent, env, project.command_line_arguments['constant_definitions'], project.command_line_arguments['prop'], project.autoencoders)
                    all_property_results.append(mdp_reward_result)

                    if (all_property_results[-1] == min(all_property_results) and prop_type == "min_prop") or (all_property_results[-1] == max(all_property_results) and prop_type == "max_prop"):
                        if project.command_line_arguments['deploy']==False and  m_autoencoder_data_collector == None:
                            project.save()
    
                    project.mlflow_bridge.log_property(all_property_results[-1], 'Property Result', episode)
                else:
                    mdp_reward_result = None
                print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", mdp_reward_result)
            else:
                # Only log reward
                all_episode_rewards.append(episode_reward)
                project.mlflow_bridge.log_reward(all_episode_rewards[-1], episode)
                try:
                    project.mlflow_bridge.log_reward_i(all_episode_rewards[-1], episode, project.agent.turn_value)
                except:
                    pass
                reward_of_sliding_window = np.mean(list(all_episode_rewards))
                project.mlflow_bridge.log_avg_reward(reward_of_sliding_window, episode)
                if len(all_property_results) > 0:
                    print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", all_property_results[-1])
                else:
                    print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", None)
                
            if project.command_line_arguments['rl_algorithm']=="turnbasedtwoagents" and prop_type=='reward' and  m_autoencoder_data_collector == None:
                if project.agent.turn_value == 0:
                    pass
                    #print("Agent 0's Turn Reward: ",episode_reward)
                elif project.agent.turn_value == 1:
                    pass
                    #print("Agent 1's Turn Reward: ",episode_reward)
                project.save()
            else:
                if reward_of_sliding_window  > best_reward_of_sliding_window and len(all_episode_rewards)>=project.command_line_arguments['sliding_window_size']:
                    best_reward_of_sliding_window = reward_of_sliding_window
                    if prop_type=='reward' and project.command_line_arguments['deploy']==False and  m_autoencoder_data_collector == None:
                        project.save()
                        print("SAVE")

            gc.collect()
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        torch.cuda.empty_cache()
        # Log overall metrics
        if project.command_line_arguments['deploy']==0 and  m_autoencoder_data_collector == None:
            project.mlflow_bridge.log_best_reward(best_reward_of_sliding_window)
            if project.command_line_arguments['task']=='safe_training':
                #TODO: Save Metrics
                pass
    
    return list(last_max_steps_states), list(last_max_steps_actions), list(last_max_steps_rewards), list(last_max_steps_terminals)


