import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import Agent
from collections import OrderedDict
import torch
import numpy as np
from common.rl_agents.deep_q_agent import DQNAgent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class CooperativeAgents(Agent):

    def __init__(self, command_line_arguments, state_dimension : int, number_of_actions : int, combined_actions, number_of_neurons : int):
        """Initialize Deep Q-Learning Agent

        Args:
            state_dimension (np.array): State Dimension
            number_of_actions (int): Number of Actions
        """
        self.number_of_actions = number_of_actions
        self.state_dimension = state_dimension
        self.combined_actions = combined_actions
        self.combined_actions.sort()
        self.number_of_neurons = number_of_neurons
        # Extract all actions for each agent (for each list element, there is a list of actions for the corresponding agent at position i)
        self.all_actions = self.extract_actions_for_agents(self.combined_actions)
        # Extract number of agents
        self.agents = []
        for i in range(len(self.all_actions)):
            self.agents.append(DQNAgent(state_dimension, number_of_neurons,len(self.all_actions[i]), epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size']))


    


    def extract_actions_for_agents(self, actions):
        """
        Extract actions for each agent.
        """
        all_actions = []
        for combined_action in actions:
            # take1_open2_close3 (number indicate agent)
            actions = combined_action.split('_')
            all_actions.append(actions)
        # Each row i contains now all the actions of agent i
        all_actions = list(zip(*all_actions))
        for i in range(len(all_actions)):
            all_actions[i] = list(all_actions[i])
            all_actions[i] = list(set(all_actions[i]))
            # Sort for each agent the actions
            all_actions[i].sort()
        return all_actions


    def local_action_index_to_name(self, agent_idx, action_index):
        """
        Convert the internal action index to the name of the action.
        """
        return self.all_actions[agent_idx][action_index]

    def local_action_name_to_idx(self, agent_idx, action_name):
        return self.all_actions[agent_idx].index(action_name)


    def combine_actions(self, actions):
        """
        Combine actions into a single action.
        """
        combined_action = ''
        for idx, action_idx in enumerate(actions):
            combined_action += self.local_action_index_to_name(idx, actions[idx]) + '_'

        selected_actions = combined_action[:-1]
        return self.combined_actions.index(selected_actions)



    def save(self):
        """
        Saves the agent onto the MLFLow Server.
        """
        for i in range(len(self.agents)):
            self.agents[i].save(artifact_path='model'+str(i))


    def load(self, model_root_folder_path):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        for idx, path in enumerate(model_root_folder_path):
            self.agents[idx].load(path)

    

    def store_experience(self, state : np.array, action : int, reward : float, n_state : np.array, done : bool):
        """Stores experience into the replay buffer

        Args:
            state (np.array): State
            action (int): action
            reward (float): reward
            n_state (np.array): next state
            done (bool): Terminal state?
        """
        combined_action_name = self.combined_actions[action]
        for agent_idx, action_name in enumerate(combined_action_name.split("_")):
            local_action_idx = self.local_action_name_to_idx(agent_idx, action_name)
            self.agents[agent_idx].store_experience(state, local_action_idx, reward, n_state, done)

            
        


    def select_action(self, state : np.ndarray, deploy=False, attack=None) -> int:
        """Select random action or action based on the current state.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, no random states. Defaults to False.

        Returns:
            [type]: action
        """
        all_action_idizes = []
        for i in range(len(self.agents)):
            action_idx = self.agents[i].select_action(state, deploy)
            all_action_idizes.append(action_idx)
        return self.combine_actions(all_action_idizes)



    def step_learn(self):
        """
        Agent learning.
        """
        for i in range(len(self.agents)):
            self.agents[i].step_learn()
   



