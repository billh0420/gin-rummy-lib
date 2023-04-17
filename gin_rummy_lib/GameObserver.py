# This version uses only GinRummyGame, DQNAgent, GinRummyNoviceRuleAgent

from collections import OrderedDict

import numpy as np

from rlcard.games.gin_rummy.game import GinRummyGame
import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils

class GameObserver:

    def __init__(self):
        pass
    
    def extract_state(self, game: GinRummyGame):
        game_legal_actions = game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in game_legal_actions}
        legal_actions = OrderedDict(legal_actions_ids)
        raw_legal_actions = list(legal_actions.keys())
        if game.is_over():
            obs = np.array([gin_rummy_utils.encode_cards([]) for _ in range(5)])
            extracted_state = {'obs': obs, 'legal_actions': legal_actions}
            extracted_state['raw_legal_actions'] = raw_legal_actions
            extracted_state['raw_obs'] = obs
        else:
            discard_pile = game.round.dealer.discard_pile
            stock_pile = game.round.dealer.stock_pile
            top_discard = [] if not discard_pile else [discard_pile[-1]]
            dead_cards = discard_pile[:-1]
            current_player = game.get_current_player()
            opponent = game.round.players[(current_player.player_id + 1) % 2]
            known_cards = opponent.known_cards
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
            hand_rep = gin_rummy_utils.encode_cards(current_player.hand)
            top_discard_rep = gin_rummy_utils.encode_cards(top_discard)
            dead_cards_rep = gin_rummy_utils.encode_cards(dead_cards)
            known_cards_rep = gin_rummy_utils.encode_cards(known_cards)
            unknown_cards_rep = gin_rummy_utils.encode_cards(unknown_cards)
            rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
            obs = np.array(rep)
            extracted_state = {'obs': obs, 'legal_actions': legal_actions, 'raw_legal_actions': raw_legal_actions}
            extracted_state['raw_obs'] = obs
        return extracted_state