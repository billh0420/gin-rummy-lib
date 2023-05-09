import numpy as np
from collections import OrderedDict

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils

class GinRummyKludge:

    def get_agent_state(self, player_id: int, game: GinRummyGame):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("GinRummyKludge get_agent_state: agent is not current player.")
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
            dead_cards = discard_pile[:-1]
            known_cards = opponent.known_cards
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
        hand_rep = utils.encode_cards(player.hand)
        top_discard_rep = utils.encode_cards(top_discard)
        dead_cards_rep = utils.encode_cards(dead_cards)
        known_cards_rep = utils.encode_cards(known_cards)
        unknown_cards_rep = utils.encode_cards(unknown_cards)
        rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
        obs = np.array(rep)
        env_state = dict()
        env_state['obs'] = obs
        env_state['agent_actions'] = agent_actions
        env_state['raw_agent_actions'] = list(agent_actions.keys())
        env_state['raw_obs'] = obs
        return env_state

    def get_agent_actions(self, player_id: int, game):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("GinRummyKludge get_legal_actions: agent is not current player.")
        legal_actions = game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)
