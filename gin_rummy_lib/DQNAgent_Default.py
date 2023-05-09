import numpy as np
import torch

import os
import pathlib

from rlcard.games.gin_rummy.utils import utils

from DQNAgent_230506 import DQNAgent_230506
from DQNAgentConfig import DQNAgentConfig

class DQNAgent_Default(DQNAgent_230506):

    # Note: Treat opponent's known cards as dead cards.

    def get_agent_state(self, player_id: int, game):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("DQNAgent_Default get_agent_state: agent is not current player.")
        agent_actions = self.get_agent_actions(player_id=player_id, game=game)
        player = game.round.players[player_id]
        opponent = game.round.players[(player_id + 1) % 2]
        stock_pile = game.round.dealer.stock_pile
        discard_pile = game.round.dealer.discard_pile
        top_discard = [] if not discard_pile else [discard_pile[-1]]
        if game.is_over():
            dead_cards = discard_pile[:-1]
            known_cards = opponent.hand
            unknown_cards = stock_pile
        else:
            dead_cards = discard_pile[:-1] + opponent.known_cards # Note this
            known_cards = [] # Note this
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in opponent.known_cards]
        hand_rep = utils.encode_cards(player.hand)
        top_discard_rep = utils.encode_cards(top_discard)
        dead_cards_rep = utils.encode_cards(dead_cards)
        known_cards_rep = utils.encode_cards(known_cards)
        unknown_cards_rep = utils.encode_cards(unknown_cards)
        rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
        obs = np.array(rep)
        agent_state = dict()
        agent_state['obs'] = obs
        agent_state['agent_actions'] = agent_actions
        agent_state['raw_agent_actions'] = list(agent_actions.keys())
        agent_state['raw_obs'] = obs
        return agent_state

    @staticmethod
    def get_dqn_agent_default(folder_path:str):
        absolute_folder_path = str(pathlib.Path(folder_path).resolve())
        model_name = "dqn_agent_default"
        agent_dir = f'{absolute_folder_path}/agents/{model_name}'
        agent_path = f'{agent_dir}/{model_name}.pth'
        config = DQNAgentConfig()
        config.model_name = model_name
        config.train_every = 5
        config.save_every = 1000000
        config.save_path = agent_dir
        if not os.path.exists(agent_path):
            agent = DQNAgent_Default(config=config)
        else:
            agent = torch.load(agent_path)
            print(f'{model_name} already exists')
        return agent
