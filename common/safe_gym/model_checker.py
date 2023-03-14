"""
This module provides the ModelChecker for the RL model checking.
"""
import json
import time
from typing import Tuple
import numpy as np
import stormpy
from stormpy.utility.utility import JsonContainerRational
from stormpy.storage.storage import SimpleValuation
import common
from common.safe_gym.permissive_manager import PermissiveManager
from common.safe_gym.abstraction_manager import AbstractionManager
from common.safe_gym.state_mapper import StateMapper
from common.adversarial_attacks.adversarial_attack_builder import AdversarialAttackBuilder
import random
import gc
import torch

class ModelChecker():
    """
    The model checker checks the product of the environment and
    the policy based on a property query.
    """

    def __init__(self, permissive_input: str, mapper: StateMapper, abstraction_input: str, attack_config_str: str):
        """Initialization

        Args:
            permissive_input (str): Permissive input for permissive model checking
            mapper (StateMapper): State Variable mapper
            abstraction_input (str): Abstraction input for state variable remapping
        """
        assert isinstance(permissive_input, str)
        assert isinstance(mapper, StateMapper)
        assert isinstance(abstraction_input, str)
        self.wrong_choices = 0
        
        self.m_permissive_manager = PermissiveManager(permissive_input, mapper)
        self.m_abstraction_manager = AbstractionManager(
            mapper, abstraction_input)

        # Attack
        attack_builder = AdversarialAttackBuilder()
        self.attack_config_str = attack_config_str
        self.mapper = mapper
        self.m_adversarial_attack = attack_builder.build_adversarial_attack(mapper, self.attack_config_str)
        # PAC stuff
        self.random_state_idx = None
        self.state_counter = 0
        self.current_state = None
        self.first_state = True
        # Benchmarking
        

    def __get_clean_state_dict(self, state_valuation_json: JsonContainerRational,
                               example_json: str) -> dict:
        """Get the clean state dictionary.

        Args:
            state_valuation_json (str): Raw state
            example_json (str): Example state as json str

        Returns:
            dict: Clean State
        """
        assert isinstance(state_valuation_json, JsonContainerRational)
        assert isinstance(example_json, str)
        state_valuation_json = json.loads(str(state_valuation_json))
        state = {}
        # print(state_valuation_json)
        # print(example_json)
        example_json = json.loads(example_json)
        for key in state_valuation_json.keys():
            for _key in example_json.keys():
                if key == _key:
                    state[key] = state_valuation_json[key]

        assert isinstance(state, dict)
        return state

    def set_pac_settings(self, random_state_idx, attack_config_str):
        self.random_state_idx = random_state_idx
        attack_builder = AdversarialAttackBuilder()
        print(random_state_idx, attack_config_str)
        self.pac_attack = attack_builder.build_adversarial_attack(self.mapper, attack_config_str)

    def __get_numpy_state(self, env, state_dict: dict) -> np.ndarray:
        """Get numpy state

        Args:
            env (SafeGym): SafeGym
            state_dict (dict): State as Dictionary

        Returns:
            np.ndarray: State
        """
        assert isinstance(state_dict, dict)
        state = env.storm_bridge.parse_state(json.dumps(state_dict))
        assert isinstance(state, np.ndarray)
        return state

    def __get_action_for_state(self, env, agent: common.rl_agents, state: np.array) -> str:
        """Get the action name for the current state

        Args:
            env (SafeGym): SafeGym
            agent (common.rl_agents): RL agents
            state (np.array): Numpy state

        Returns:
            str: Action name
        """
        assert str(agent.__class__).find("common.rl_agents") != -1
        #assert isinstance(state, np.ndarray)
        # PAC
        if self.random_state_idx is not None:
            if self.first_state:
                self.current_state = np.array(state, copy=True)
                self.first_state = False
            if np.array_equal(state, self.current_state) and self.first_state == False:
                # Random Attack
                attack_builder = AdversarialAttackBuilder()
                attack_config_str = "fgsm,"+str(random.uniform(0,0.1))
                self.m_adversarial_attack = attack_builder.build_adversarial_attack(self.mapper, attack_config_str)
            else:
                self.current_state = np.array(state, copy=True)
                self.m_adversarial_attack = None
                self.state_counter+=1
        # pass attack to agent (for MARL)
        action_index = agent.select_action(state, True, attack=self.m_adversarial_attack)
        action_name = env.action_mapper.actions[action_index]
        assert isinstance(action_name, str)
        return action_name

    def induced_markov_chain(self, agent: common.rl_agents, env,
                             constant_definitions: str,
                             formula_str: str, autoencoders = None) -> Tuple[float, int]:
        """Creates a Markov chain of an MDP induced by a policy
        and applies model checking.py

        Args:
            agent (common.rl_agents): RL policy
            env (SafeGym): SafeGym
            constant_definitions (str): Constant definitions
            formula_str (str): Property query

        Returns:
            Tuple: Tuple of the property result, model size and performance metrices
        """
        assert str(agent.__class__).find("common.rl_agents") != -1
        assert isinstance(constant_definitions, str)
        assert isinstance(formula_str, str)
        
        first_state = True
        env.reset()
        self.m_permissive_manager.action_mapper = env.action_mapper
        self.wrong_choices = 0
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(env.storm_bridge.path)
        suggestions = dict()
        i = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = "tau_" + \
                        str(i)  # str(m.name)
                    i += 1

        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], constant_definitions)[0].as_prism_program()

        prism_program = prism_program.label_unlabelled_commands(suggestions)

        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        #options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        options.set_build_choice_labels(True)

        def permissive_policy(state_valuation: SimpleValuation, action_index: int) -> bool:
            """Whether for the given state and action, the action should be allowed in the model.

            Args:
                state_valuation (SimpleValuation): State valuation
                action_index (int): Action index

            Returns:
                bool: Allowed?
            """
            assert isinstance(state_valuation, SimpleValuation)
            assert isinstance(action_index, int)
            simulator.restart(state_valuation)
            available_actions = sorted(simulator.available_actions())
            action_name = prism_program.get_action_name(action_index)
            # conditions on the action
            state = self.__get_clean_state_dict(
                state_valuation.to_json(), env.storm_bridge.state_json_example)
            try:
                turn = state['turn']
            except:
                pass
            state = self.__get_numpy_state(env, state)
            #print(state)
            # Attack State and Denoise state with autoencoder (if no attack, only denoise)
            clean_obs = []
            if autoencoders is not None and len(autoencoders) !=0:
                #print("Original", state)
                for i in range(len(autoencoders)):
                    tmp_ob = autoencoders[i].attack_and_clean(state, agent, i)
                    #print(tmp_ob)
                    clean_obs.append(tmp_ob.detach().cpu().numpy())
                    #print(state)
                #print("Cleaned",state)
                state = clean_obs
                


            
            # State Abstraction
            if self.m_abstraction_manager.is_active:
                state = self.m_abstraction_manager.preprocess_state(state)
            

            # Check if selected action is available..
            # if not set action to the first available action
            if len(available_actions) == 0:
                return False

            cond1 = False
            if self.m_permissive_manager.is_permissive:
                self.m_permissive_manager.manage_actions(state, agent)
                cond1 = self.m_permissive_manager.create_condition(
                    available_actions, action_name)
            else:
                selected_action = self.__get_action_for_state(
                    env, agent, state)
                if (selected_action in available_actions) is not True:
                    selected_action = available_actions[0]
                #print(state, selected_action)
                cond1 = (action_name == selected_action)
            
            
            #torch.cuda.empty_cache()
            #gc.collect()
            # print(str(state_valuation.to_json()), action_name)#, state, selected_action, cond1)
            assert isinstance(cond1, bool)
            return cond1

        model_building_start = time.time()
        simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(
            stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

        constructor = stormpy.make_sparse_model_builder(prism_program, options,
                                                        stormpy.StateValuationFunctionActionMaskDouble(
                                                            permissive_policy))
        model = constructor.build()
        model_size = len(model.states)
        print("Model Building Time:", time.time()-model_building_start)
        print("Model Size:", model.nr_states)
        print("Transitions", model.nr_transitions)
        print("Attack Config (empty=No Attack):", self.attack_config_str)
        # print(model)
        #formula_str = formula_str.replace("min", "max")
        print(formula_str)
        model_checking_start_time = time.time()
        print("Parse Properties...")
        properties = stormpy.parse_properties(formula_str, prism_program)
        print("Model Cheking...")
        result = stormpy.model_checking(model, properties[0])
        print("Model Checking Time:", time.time()-model_checking_start_time)
        
        stormpy.export_to_drn(model,"test.drn")
        initial_state = model.initial_states[0]
        #print('Result for initial state', result.at(initial_state))
        mdp_reward_result = result.at(initial_state)
        # Update StateActionCollector
        assert isinstance(mdp_reward_result, float)
        assert isinstance(model_size, int)
        return mdp_reward_result, model_size
