import numpy as np
from collections import OrderedDict

from rlcard.agents import DQNAgent
from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils

from DQNAgentConfig import DQNAgentConfig
from rlcard.games.gin_rummy.utils.scorers import knock_action_id

class DQNAgent_Gins(DQNAgent):

    def __init__(self, config: DQNAgentConfig):
        replay_memory_size = 20000
        replay_memory_init_size = 100
        update_target_estimator_every = 1000
        discount_factor = 0.99
        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay_steps = 20000
        batch_size = 32
        num_actions = 2
        state_shape = None
        train_every = 1
        mlp_layers = None
        learning_rate = 0.00005
        device = None
        save_every = float('inf')
        model_name = 'dqn_agent'
        for key, value in config.to_dict().items():
            print(f'{key}: {value}')
            if key == 'replay_memory_size':
                replay_memory_size = value
            elif key == 'replay_memory_init_size':
                replay_memory_init_size = value
            elif key == 'update_target_estimator_every':
                update_target_estimator_every = value
            elif key == 'discount_factor':
                discount_factor = value
            elif key == 'epsilon_start':
                epsilon_start = value
            elif key == 'epsilon_end':
                epsilon_end = value
            elif key == 'epsilon_decay_steps':
                epsilon_decay_steps = value
            elif key == 'batch_size':
                batch_size = value
            elif key == 'num_actions':
                num_actions = value
            elif key == 'state_shape':
                state_shape = value
            elif key == 'train_every':
                train_every = value
            elif key == 'mlp_layers':
                mlp_layers = value
            elif key == 'learning_rate':
                learning_rate = value
            elif key == 'device':
                device = value
            elif key == 'save_every':
                save_every = value
            elif key == 'model_name':
                model_name = value

        save_path = None

        super().__init__(replay_memory_size,
                         replay_memory_init_size,
                         update_target_estimator_every,
                         discount_factor,
                         epsilon_start,
                         epsilon_end,
                         epsilon_decay_steps,
                         batch_size,
                         num_actions,
                         state_shape,
                         train_every,
                         mlp_layers,
                         learning_rate,
                         device,
                         save_path,
                         save_every)

    def get_env_state(self, player_id: int, game: GinRummyGame):
        legal_actions = self.get_legal_actions(game=game)
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
        env_state = dict()
        env_state['obs'] = obs
        env_state['legal_actions'] = legal_actions
        env_state['raw_legal_actions'] = list(legal_actions.keys())
        env_state['raw_obs'] = obs
        return env_state

    def get_legal_actions(self, game: GinRummyGame):
        legal_actions = game.judge.get_legal_actions()
        legal_actions = [x for x in legal_actions if x.action_id < knock_action_id] # Note: omit knock actions
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)
