from common.adversarial_attacks.adversarial_attack import AdversarialAttack
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os
from numpy import linalg as LA
import random

class RandomAttack(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        attack_name, epsilon = attack_config_str.split(',')
        self.attack_name = attack_name
        self.epsilon = float(epsilon)

    def generate_random_attack(self, state):
        tmp_epsilon = self.epsilon
        tmp_attack = np.zeros(state.shape[0])
        # Create a random numpy array
        for i in range(state.shape[0]):
            rnd_value = random.uniform(0,self.epsilon)
            tmp_attack[i]=rnd_value
            if random.randint(0,state.shape[0])==0:
                np.random.shuffle(tmp_attack)
                return state + tmp_attack, tmp_attack
            #print(tmp_epsilon)
        np.random.shuffle(tmp_attack)
        return state + tmp_attack, tmp_attack


        


    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        if self.already_attacked(state):
            #print("Already", state + self.attack_buffer[str(state)])
            return state + self.attack_buffer[str(state)]
        else:
            adversarial_state, adv_perturbation = self.generate_random_attack(state)
            self.update_attack_buffer(state, adv_perturbation)
            self.save_max_l1_norm(adv_perturbation)
            #print(adversarial_state)
            return adversarial_state