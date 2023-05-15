import numpy as np
from collections import OrderedDict

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils
from rlcard.games.gin_rummy.utils.action_event import ActionEvent, ScoreNorthPlayerAction, ScoreSouthPlayerAction

from dqn_agent_230510.DQNAgent import DQNAgent

class DQNAgent_AnyAction(DQNAgent):

    # Note: Treat opponent's known cards as dead cards.

    def get_agent_state(self, player_id: int, game):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("DQNAgent_AnyAction get_agent_state: agent is not current player.")
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

    def get_agent_actions(self, player_id: int, game: GinRummyGame):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("DQNAgent_AnyAction get_legal_actions: agent is not current player.")
        legal_actions = game.judge.get_legal_actions()
        if not legal_actions:
            pass
        elif legal_actions[0] is ScoreNorthPlayerAction or legal_actions[0] is ScoreSouthPlayerAction:
            pass
        else:
            legal_actions = [ActionEvent.decode_action(x) for x in range(110)]
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)
