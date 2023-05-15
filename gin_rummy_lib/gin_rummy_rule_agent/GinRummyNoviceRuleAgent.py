from typing import List

from collections import OrderedDict

import numpy as np

from rlcard.games.base import Card

from rlcard.games.gin_rummy.utils.action_event import ActionEvent, DiscardAction, KnockAction, GinAction
import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils
import rlcard.games.gin_rummy.utils.melding as melding

from GinRummyAgentStateMixin import GinRummyAgentStateMixin
from GinRummyAgentActionsMixin import GinRummyAgentActionsMixin
from GameAgent import GameAgent

class GinRummyNoviceRuleAgent(GinRummyAgentStateMixin, GinRummyAgentActionsMixin, GameAgent):
    '''
        Agent always discards highest deadwood value card
    '''

    def __init__(self):
        super().__init__()
        self.use_raw = False  # FIXME: should this be True ?

    def eval_step(self, agent_state):
        ''' Predict the action given the current state.
            Novice strategy:
                Case where can gin:
                    Choose one of the gin actions.
                Case where can knock:
                    Choose one of the knock actions.
                Case where can discard:
                    Gin if can. Knock if can.
                    Otherwise, put aside cards in some best meld cluster.
                    Choose one of the remaining cards with highest deadwood value.
                    Discard that card.
                Case otherwise:
                    Choose a random action.

        Args:
            agent_state (numpy.array): an numpy array that represents the current state

        Returns:
            action_id (int): the action predicted
        '''
        legal_actions = agent_state['agent_actions']
        actions = legal_actions.copy()
        legal_action_events = [ActionEvent.decode_action(x) for x in legal_actions]
        gin_action_events = [x for x in legal_action_events if isinstance(x, GinAction)]
        knock_action_events = [x for x in legal_action_events if isinstance(x, KnockAction)]
        discard_action_events = [x for x in legal_action_events if isinstance(x, DiscardAction)]
        if gin_action_events:
            actions = [x.action_id for x in gin_action_events]
        elif knock_action_events:
            actions = [x.action_id for x in knock_action_events]
        elif discard_action_events:
            best_discards = self.get_best_discards(discard_action_events=discard_action_events, agent_state=agent_state)
            if best_discards:
                actions = [DiscardAction(card=card).action_id for card in best_discards]
        if type(actions) == OrderedDict:
            actions = list(actions.keys())
        return np.random.choice(actions)

    def get_best_discards(self, discard_action_events, agent_state) -> List[Card]:
        best_discards: List[Card] = []
        final_deadwood_count = 999
        env_hand = agent_state['obs'][0]
        hand = gin_rummy_utils.decode_cards(env_cards=env_hand)
        for discard_action_event in discard_action_events:
            discard_card = discard_action_event.card
            next_hand = [card for card in hand if card != discard_card]
            meld_clusters = melding.get_meld_clusters(hand=next_hand)
            deadwood_counts = []
            for meld_cluster in meld_clusters:
                deadwood_count = gin_rummy_utils.get_deadwood_count(hand=next_hand, meld_cluster=meld_cluster)
                deadwood_counts.append(deadwood_count)
            best_deadwood_count = min(deadwood_counts,
                                      default=gin_rummy_utils.get_deadwood_count(hand=next_hand, meld_cluster=[]))
            if best_deadwood_count < final_deadwood_count:
                final_deadwood_count = best_deadwood_count
                best_discards = [discard_card]
            elif best_deadwood_count == final_deadwood_count:
                best_discards.append(discard_card)
        return best_discards
