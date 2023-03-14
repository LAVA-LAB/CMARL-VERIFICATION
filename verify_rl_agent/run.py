import sys
sys.path.insert(0, '../')
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *





if __name__ == '__main__':
    command_line_arguments = get_arguments()
    run_verify_rl_agent(command_line_arguments)
    
