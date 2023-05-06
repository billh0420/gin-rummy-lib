import numpy as np

from rlcard.games.gin_rummy.utils import utils

from DQNAgent_230506 import DQNAgent_230506

class DQNAgent_Default(DQNAgent_230506):

    # Note: Treat opponent's known cards as dead cards.

    def get_env_state(self, player_id: int, game):
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
