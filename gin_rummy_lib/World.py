import os
import torch
import pandas as pd
import panel as pn

from rlcard.utils import get_device
from rlcard.agents import DQNAgent

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from GinRummyRookie01RuleAgent import GinRummyRookie01RuleAgent
from GinRummyLoserRuleAgent import GinRummyLoserRuleAgent
from rlcard.agents.random_agent import RandomAgent

from util import get_current_time
from util import game_settings_to_dict
from RLTrainerConfig import RLTrainerConfig
from RLTrainer import RLTrainer
from GameMaker import GameMaker

class World:

    def __init__(self, game_maker:GameMaker, world_dir:str or None = None):
        self.game_maker = game_maker
        self.world_dir = world_dir if world_dir else os.path.abspath('.')

        self.model_name = None # determines the training agent
        self.model_name = 'dqn_agent' # FIXME: temp kludge

        # Define configs
        self.rl_trainer_config = RLTrainerConfig()

        # More stuff
        num_actions = self.get_game_num_actions()
        self.opponent_agent = RandomAgent(num_actions=num_actions)
        self.opponent_agent = GinRummyNoviceRuleAgent()
        self.opponent_agent = GinRummyRookie01RuleAgent()
        self.opponent_agent = GinRummyLoserRuleAgent()

    @property
    def agent(self):
        result = None
        agent_path = self.agent_path
        if agent_path and os.path.exists(agent_path):
            result = torch.load(agent_path)
        return result

    @property
    def opponents(self):
        opponent_agent = self.opponent_agent
        if not opponent_agent:
            opponent_agent = GinRummyNoviceRuleAgent()
        return [opponent_agent]

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

    @property
    def game_settings(self):
        return self.game_maker.make_game().settings

    @property
    def game_settings_dict(self):
        return game_settings_to_dict(settings=self.game_settings)

    def get_game_num_actions(self) -> int:
        game = self.game_maker.make_game()
        num_actions = game.get_num_actions()
        return num_actions

    def play_train_match(self, num_episodes: int or None = None):
        agent_dir = self.agent_dir
        agent = self.agent
        if agent and agent_dir:
            game = self.game_maker.make_game()
            rl_trainer_config = self.rl_trainer_config
            # Print current configuration
            print("Starting training")
            game.settings.print_settings()
            print(f'actual scorer_name={game.judge.scorer.name}')
            print(f'==============================')
            print(f"Start: {get_current_time()}")

            print(f'----- DQN Agent Config -----')
            for key, value in self.dqn_agent_to_dict(agent).items():
                print(f'{key}: {value}')

            print(self.rl_trainer_config)
            print(f'train_steps={agent.train_t} time_steps={agent.total_t}')
            print('----- agent.q_estimator.qnet -----')
            print(agent.q_estimator.qnet)
            # train agent
            actual_num_episodes = num_episodes if num_episodes else self.rl_trainer_config.num_episodes
            rlTrainer = RLTrainer(
                game=game,
                agent=agent,
                opponents=self.opponents,
                log_dir=agent_dir,
                model_name= self.model_name,
                rl_trainer_config=rl_trainer_config)
            rlTrainer.train(num_episodes=actual_num_episodes)
        else:
            print("You need to select a dqn_agent")

    def dqn_agent_to_dict(self, dqn_agent):
        result = dict()
        result['replay_memory_size'] = dqn_agent.memory.memory_size
        result['replay_memory_init_size'] = dqn_agent.replay_memory_init_size
        result['update_target_estimator_every'] = dqn_agent.update_target_estimator_every
        result['discount_factor'] = dqn_agent.discount_factor
        result['epsilon_start'] = dqn_agent.epsilons[0]
        result['epsilon_end'] = dqn_agent.epsilons[-1]
        result['epsilon_decay_steps'] = dqn_agent.epsilon_decay_steps
        result['batch_size'] = dqn_agent.batch_size
        result['train_every'] = dqn_agent.train_every
        result['save_every'] = dqn_agent.save_every
        result['learning_rate'] = dqn_agent.q_estimator.learning_rate
        result['num_actions'] = dqn_agent.q_estimator.num_actions
        result['state_shape'] = dqn_agent.q_estimator.state_shape
        result['mlp_layers'] = dqn_agent.q_estimator.mlp_layers
        # result['model_name'] = dqn_agent.model_name
        return result