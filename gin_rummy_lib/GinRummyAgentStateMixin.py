import numpy as np

from rlcard.games.gin_rummy.game import GinRummyGame

import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils

class GinRummyAgentStateMixin:

    def get_agent_state(self, player_id: int, game: GinRummyGame):
        ''' Encode state

        Args:
            player_id: int
            game: GinRummyGame

        Returns:
            numpy array: 5 * 52 array
                         5 : current hand (1 if card in hand else 0)
                             top_discard (1 if card is top discard else 0)
                             dead_cards (1 for discards except for top_discard else 0)
                             opponent known cards (likewise)
                             unknown cards (likewise)  # is this needed ??? 200213
        '''
    def get_agent_state(self, player_id: int, game: GinRummyGame):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("GinRummyAgent get_agent_state: agent is not current player.")
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
        hand_rep = gin_rummy_utils.encode_cards(player.hand)
        top_discard_rep = gin_rummy_utils.encode_cards(top_discard)
        dead_cards_rep = gin_rummy_utils.encode_cards(dead_cards)
        known_cards_rep = gin_rummy_utils.encode_cards(known_cards)
        unknown_cards_rep = gin_rummy_utils.encode_cards(unknown_cards)
        rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
        obs = np.array(rep)
        agent_state = dict()
        agent_state['obs'] = obs
        agent_state['agent_actions'] = agent_actions
        agent_state['raw_agent_actions'] = list(agent_actions.keys())
        agent_state['raw_obs'] = obs
        return agent_state
