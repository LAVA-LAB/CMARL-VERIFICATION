import sys
sys.path.insert(0, '../')
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import random
from helper import *
from math import *

# Experiments
# For tic-tac-toe
# Load player 2 from another training process and apply model checking of current player 1 against new player 2
# Do the same for multiple player 2s with different architecture parameters
# Calculate the average of the results of the model checking and compare with pac guarantees

if __name__ == '__main__':
    start_time = time.time()
    start_timestemp = get_current_timestemp()
    command_line_arguments = get_arguments()
    threshold = command_line_arguments['epsilon']
    alpha = command_line_arguments['alpha']
    outputs = run_verify_rl_agent(command_line_arguments)
    result = outputs[0]
    N = command_line_arguments['range_plotting']
    print("Result", result)
    model_size = outputs[1]
    print("Model size", model_size)
    all_results = []
    for i in range(N):
        rand_state_idx = random.randint(0, model_size -1)
        random_epsilon =  random.uniform(0,1)
        outputs = run_verify_rl_agent(command_line_arguments,random_state_idx=rand_state_idx, random_epsilon=random_epsilon)
        clean_folder("../mlruns",start_timestemp)
        print("Current Sample IDX:", i)
        if abs(result-outputs[0]) <= threshold:
            all_results.append(1)
        else:
            all_results.append(0)
        #all_results.append(outputs[0])
    print("Average Result:", sum(all_results)/N)
    g = exp(-N *(alpha**2)/2) + exp(-N *(alpha**2)/3)
    print(f"Probability that we are further away from the Expected Result than {alpha}:", g)
    print("Running time:", time.time() - start_time)
        
    
