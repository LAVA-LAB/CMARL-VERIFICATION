import torch
from torch.utils.data import Dataset
from os import walk
import os
import numpy as np
import mlflow
import shutil
from common.adversarial_attacks.fgsm import FGSM
import random
import getpass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NORM_SCALE = 40
HIDDEN_LAYER_SIZE = 1048


class AEDataset(Dataset):

    def __init__(self, root_dir, coop_agent, agent_idx):
        self.root_dir = root_dir
        self.coop_agent = coop_agent
        self.agent_idx = agent_idx
        self.file_paths = []
        for (dirpath, dirnames, filenames) in walk(root_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    self.file_paths.append(os.path.join(dirpath, filename))
        self.create_adv_data()

    def artificial_data_generation(self, n):
        """
        Sample from self.file_path and add a random noise to it
        """
        print("Generate",n," samples.")
        for i in range(n):
            for i in range(3):
                random_file_path = random.sample(self.file_paths, 1)[0]
                x = np.load(random_file_path)
                x = self.coop_agent.po_manager.get_observation(x, self.agent_idx)
                # Random numpy noise
                np.random.shuffle(x)
                rnd_epsilon = random.random()
                rnd_epsilon = 0.1
                m_fgsm = FGSM(self.coop_agent.po_manager.state_mapper, "fgsm,"+str(rnd_epsilon))
                adv_data = m_fgsm.attack(self.coop_agent.agents[self.agent_idx], x)
                adv_data = torch.from_numpy(adv_data)
                #print(adv_data)
                self.X.append(adv_data/NORM_SCALE)
                self.y.append(x/NORM_SCALE)
                self.X.append(x/NORM_SCALE)
                self.y.append(x/NORM_SCALE)


    def create_adv_data(self):
        self.X = []
        self.y = []
        for idx, file_path in enumerate(self.file_paths):
            # Read npy
            x = np.load(file_path)
            x = self.coop_agent.po_manager.get_observation(x, self.agent_idx)
            # numpy to pytorch tensor
            #x = torch.from_numpy(x)
            # Create adversarial data
            rnd_epsilon = random.random()
            rnd_epsilon = 0.1
            m_fgsm = FGSM(self.coop_agent.po_manager.state_mapper, "fgsm,"+str(rnd_epsilon))
            adv_data = m_fgsm.attack(self.coop_agent.agents[self.agent_idx], x)
            adv_data = torch.from_numpy(adv_data)
            #print(adv_data)
            self.X.append(adv_data/NORM_SCALE)
            self.y.append(x/NORM_SCALE)
            self.X.append(x/NORM_SCALE)
            self.y.append(x/NORM_SCALE)
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))


class AE(torch.nn.Module):

    def __init__(self, input_output_size):
        super().__init__()
        self.input_output_size = input_output_size
         

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_output_size, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, input_output_size),
            torch.nn.ReLU()
        )
        self.epsilon = None
        self.to(device)
 
    def forward(self, x):
        x = self.nn(x)
        return x

    def attack_and_clean(self, x, rl_agent, agent_idx):
        # Get first observation of agent
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        if self.epsilon == None:
            obs = torch.from_numpy(obs/NORM_SCALE).to(device)
            clean_obs = torch.round(self.forward(obs.float())*NORM_SCALE)
            #x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            return clean_obs
        else:
            m_fgsm = FGSM(rl_agent.po_manager.state_mapper, "fgsm,"+str(self.epsilon))
            #print(obs)
            adv_data = m_fgsm.attack(rl_agent.agents[agent_idx], obs)
            #print("adv_data", adv_data)
            adv_data = torch.from_numpy(adv_data/NORM_SCALE).to(device)
            clean_obs = torch.round(self.forward(adv_data)*NORM_SCALE)
            #print("Cleaned", clean_obs)
            #x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            #print("Injected", x)
            return clean_obs

    def attack(self, x, rl_agent, agent_idx):
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        if self.epsilon != None:
            m_fgsm = FGSM(rl_agent.po_manager.state_mapper, "fgsm,"+str(self.epsilon))
            adv_data = m_fgsm.attack(rl_agent.agents[agent_idx], obs)
            return adv_data
        else:
            return obs

    def clean(self, x, rl_agent, agent_idx):
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        obs = torch.from_numpy(obs/NORM_SCALE).to(device)
        clean_obs = torch.round(self.forward(obs.float())*NORM_SCALE)
        return clean_obs



    
    def set_attack(self, attack:str):
        self.attack_name, epsilon = attack.split(",")
        self.epsilon = float(epsilon)
        

    def save(self, idx):
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            username = getpass.getuser()
        torch.save(self.nn.state_dict(), 'tmp_model/nn_'+str(self.input_output_size)+'_.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path="autoencoder" + str(idx))
        shutil.rmtree('tmp_model')


    def load(self, model_path, idx):
        # replace last foldername with autoencoder of the model_path
        model_path = model_path[0].replace("model", "autoencoder" + str(idx))
        self.nn.load_state_dict(torch.load(os.path.join(model_path, 'nn_'+str(self.input_output_size)+'_.chkpt')))