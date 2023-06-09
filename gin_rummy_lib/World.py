import os
import torch
import pandas as pd
import panel as pn
import pathlib
import glob

from rlcard.utils import get_device
from rlcard.agents import DQNAgent

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings
from gin_rummy_rule_agent.GinRummyNoviceRuleAgent import GinRummyNoviceRuleAgent
from gin_rummy_rule_agent.GinRummyRookie01RuleAgent import GinRummyRookie01RuleAgent
from gin_rummy_rule_agent.GinRummyLoserRuleAgent import GinRummyLoserRuleAgent
from rlcard.agents.random_agent import RandomAgent

from util import get_current_time
from RLTrainerConfig import RLTrainerConfig
from RLTrainer230506 import RLTrainer230506

from pane.DQNAgentPane import DQNAgentPane

class World:

    def __init__(self, game: GinRummyGame, world_dir:str or None = None):
        self.game = game
        self.world_dir = world_dir if world_dir else os.path.abspath('.')

        self.model_name = None # determines the training agent
        self.model_name = 'dqn_agent' # FIXME: temp kludge

        # Define configs
        self.rl_trainer_config = RLTrainerConfig()

        # opponents and current opponent
        self.opponent_names = ['Random', 'Novice', 'Rookie01', 'Loser']
        self.opponent_name = 'Novice'

    @property
    def opponent(self):
        opponent = GinRummyNoviceRuleAgent()
        opponent_name = self.opponent_name
        num_actions = self.game.get_num_actions()
        if opponent_name == 'Random':
            opponent = RandomAgent(num_actions=num_actions)
        elif opponent_name == 'Novice':
            opponent = GinRummyNoviceRuleAgent()
        elif opponent_name == 'Rookie01':
            opponent = GinRummyRookie01RuleAgent()
        elif opponent_name == 'Loser':
            opponent = GinRummyLoserRuleAgent()
        return opponent

    @property
    def agent(self):
        result = None
        agent_path = self.agent_path
        if agent_path and os.path.exists(agent_path):
            result = torch.load(agent_path)
        return result

    @property
    def opponents(self):
        opponent = self.opponent
        if not opponent:
            opponent = GinRummyNoviceRuleAgent()
        return [opponent]

    @property
    def agent_dir(self):
        result = None
        agent_path = self.agent_path
        if agent_path:
            result = os.path.dirname(agent_path)
        return result

    @property
    def agent_path(self):
        result = None
        model_name = self.model_name
        if model_name:
            result = f'{self.world_dir}/agents/{model_name}/{model_name}.pth'
        return result

    def play_train_match(self, num_episodes: int or None = None):
        agent_dir = self.agent_dir
        agent = self.agent
        if agent and agent_dir:
            game = self.game
            rl_trainer_config = self.rl_trainer_config
            # Print current configuration
            print("Starting training")
            game.settings.print_settings()
            print(f'actual scorer_name={game.judge.scorer.name}')
            print(f'==============================')
            print(f"Start: {get_current_time()}")

            print(f'----- DQN Agent Config -----')
            for key, value in DQNAgentPane.dqn_agent_to_dict(agent).items():
                print(f'{key}: {value}')

            print(self.rl_trainer_config)
            print(f'train_steps={agent.train_t} time_steps={agent.total_t}')
            print('----- agent.q_estimator.qnet -----')
            print(agent.q_estimator.qnet)
            # train agent
            agents = [agent] + self.opponents
            actual_num_episodes = num_episodes if num_episodes is not None else self.rl_trainer_config.num_episodes
            rlTrainer = RLTrainer230506()
            rlTrainer.train(agents, game=game, num_episodes=num_episodes, num_eval_games=rl_trainer_config.num_eval_games)
        else:
            print("You need to select a dqn_agent")

    @staticmethod
    def get_absolute_path_for_folder(folder, search_root: str = '/content'): # default '/content' for google colab
        result = None
        possible_local_path = glob.glob(f'{search_root}/**/{folder}', recursive=True)
        if len(possible_local_path) == 1:
            result = pathlib.Path(possible_local_path[0]).resolve()
        return result

    @staticmethod
    def enter_folder(folder:str, can_mkdir = True):
        absolute_path = World.get_absolute_path_for_folder(folder=folder)
        if absolute_path:
            os.chdir(absolute_path)
        elif can_mkdir:
            os.mkdir(folder)
            os.chdir(folder)
        else:
            print(f'No such folder: {folder}')
