import numpy as np

from collections import OrderedDict

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils
from rlcard.games.gin_rummy.utils.move import ScoreSouthMove

from Env2 import Env2

class GinRummyEnv2(Env2):

    def __init__(self):
        self.name = 'gin-rummy'
        self.game = GinRummyGame()
        self.num_players = self.game.get_num_players()
        self._ScoreSouthMove = ScoreSouthMove
        super().__init__(config={'allow_step_back': False, 'seed': None})

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        # determine whether game completed all moves
        is_game_complete = False
        if self.game.round:
            move_sheet = self.game.round.move_sheet
            if move_sheet and isinstance(move_sheet[-1], self._ScoreSouthMove):
                is_game_complete = True
        payoffs = [0, 0] if not is_game_complete else self.game.judge.scorer.get_payoffs(game=self.game)
        return np.array(payoffs)

    def _decode_action(self, action_id):  # FIXME 200213 should return str
        ''' Action id -> the action in the game. Must be implemented in the child class.
        Args:
            action_id (int): the id of the action
        Returns:
            action (ActionEvent): the action that will be passed to the game engine.
        '''
        return self.game.decode_action(action_id=action_id)

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = self.game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)

    def _extract_state(self, state):  # 200213 don't use state ???
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            numpy array
        '''
        if self.game.is_over():
            obs = np.array([utils.encode_cards([]) for _ in range(5)])
            legal_actions = self._get_legal_actions()
            extracted_state = {'obs': obs, 'legal_actions': legal_actions}
            extracted_state['raw_legal_actions'] = list(legal_actions.keys())
            extracted_state['raw_obs'] = obs
        else:
            discard_pile = self.game.round.dealer.discard_pile
            stock_pile = self.game.round.dealer.stock_pile
            top_discard = [] if not discard_pile else [discard_pile[-1]]
            dead_cards = discard_pile[:-1]
            current_player = self.game.get_current_player()
            opponent = self.game.round.players[(current_player.player_id + 1) % 2]
            known_cards = opponent.known_cards
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
            hand_rep = utils.encode_cards(current_player.hand)
            top_discard_rep = utils.encode_cards(top_discard)
            dead_cards_rep = utils.encode_cards(dead_cards)
            known_cards_rep = utils.encode_cards(known_cards)
            unknown_cards_rep = utils.encode_cards(unknown_cards)
            rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
            obs = np.array(rep)
            legal_actions = self._get_legal_actions()
            extracted_state = {'obs': obs, 'legal_actions': legal_actions, 'raw_legal_actions': list(legal_actions.keys())}
            extracted_state['raw_obs'] = obs
        return extracted_state