
import sys
sys.path.insert(0, '../')
from common.tasks.safe_gym_training import *
from common.utilities.project import Project
from common.autoencoders.autoencoder import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from os import walk
import gc
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    command_line_arguments = get_arguments()
    artificial_samples = command_line_arguments['replay_buffer_size']
    state_descriptions = command_line_arguments['permissive_input']
    if command_line_arguments['parent_run_id']=="":
        print("No parent run id provided. Exiting.")
        exit(-1)
    command_line_arguments["autoencoder_folder"] = command_line_arguments['prism_file_path'].split(".")[0]+"_data"
    command_line_arguments['deploy'] = 1
    if os.path.exists(command_line_arguments["autoencoder_folder"]) == False:
        os.mkdir(command_line_arguments["autoencoder_folder"])
    set_random_seed(command_line_arguments['seed'])
    
    experiment_id = run_safe_gym_training(command_line_arguments)
    print("Experiment ID:", experiment_id)    
    # Load Project
    env = SafeGym(os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path']), command_line_arguments['constant_definitions'], 
                command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'],
                  command_line_arguments['reward_flag'], 
                  command_line_arguments['seed'], command_line_arguments['permissive_input'],
                  command_line_arguments['disabled_features'], command_line_arguments['attack_config'])
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments,
                           env.observation_space, env.action_space, all_actions=env.action_mapper.actions)
    m_project.agent.load_env(env)
    print("Project loaded",m_project.command_line_arguments)
    # For which agent do we want to have an autoencoder? -> For all
    autoencoders = []
    # For each autoencoder
    for i in range(len(m_project.agent.agents)):
        torch.manual_seed(860523297119962652)
        #860523297119962652
        print(torch.seed())
        # For each state
        m_dataset = AEDataset(command_line_arguments["autoencoder_folder"], m_project.agent, i)
        m_dataset.artificial_data_generation(artificial_samples)
        m_data_loader = DataLoader(dataset=m_dataset, batch_size=32, shuffle=True)
        # Train Autoencoder
        autoencoders.append(AE(m_project.agent.po_manager.get_observation_dimension_for_agent_idx(i)))
        loss_function = torch.nn.MSELoss()
 
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(autoencoders[i].parameters(),
                                    lr = 0.0001)
        # Plot
        epochs = command_line_arguments['eval_interval']
        print("Training Autoencoder for agent",i)
        losses = []
        for epoch in range(epochs):
            epoch_loss = []
            for (images, original_images) in m_data_loader:
                images = images.to(device).float()

                # Output of Autoencoder
                reconstructs = autoencoders[i](images)
                # numpy array to pytorch tensor
                original_images = original_images.float()


                # Original image to pytorch tensor on gpu
                original_images = original_images.to(device)
            
                # Calculating the loss function
                loss = loss_function(reconstructs, original_images)
                
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Storing the losses in a list for plotting
                losses.append(loss.cpu().detach().numpy())
                epoch_loss.append(losses[-1])
            print(epoch, "Average Epoch Loss",sum(epoch_loss)/len(epoch_loss))
            m_project.mlflow_bridge.log_denoiser_loss(i, sum(epoch_loss)/len(epoch_loss),epoch)
            torch.cuda.empty_cache()
            gc.collect()
        
        # Defining the Plot Style
        #plt.style.use('fivethirtyeight')
        #plt.xlabel('Iterations')
        #plt.ylabel('Loss')
        
        # Plotting the last 100 values
        #plt.plot(losses[-100:])
        #plt.show()
        # Save autoencoder to project copy
        
        
    # Save project copy
    m_project.save()
    m_project.save_autoencoders(autoencoders)