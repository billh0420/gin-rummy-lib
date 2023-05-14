# Class GinRummyRookie01RuleAgent
#
# This is currently the strongest rule agent for Gin Rummy.

from typing import List

import numpy as np

from rlcard.games.base import Card
from rlcard.games.gin_rummy.utils.action_event import ActionEvent
from rlcard.games.gin_rummy.utils.action_event import DiscardAction, KnockAction, GinAction
from rlcard.games.gin_rummy.utils.action_event import pick_up_discard_action_id
from rlcard.games.gin_rummy.utils.thinker import Thinker

import rlcard.games.gin_rummy.utils.melding as melding
import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils

from GinRummyAgentStateMixin import GinRummyAgentStateMixin
from GinRummyAgentActionsMixin import GinRummyAgentActionsMixin
from GameAgent import GameAgent

class GinRummyRookie01RuleAgent(GinRummyAgentStateMixin, GinRummyAgentActionsMixin, GameAgent):
    '''
        Always gin if can.
        Always knock if can.
        Always discards highest deadwood value card.
        Pick up discard card if it forms a worthwhile meld else draw a card (or declare dead hand if cannot draw).
    '''

    def __init__(self):
        super().__init__()
        self.use_raw = False

    def eval_step(self, agent_state) -> int:
        ''' Predict the action_id given the current state for evaluation.
            Since the agents is not trained, this function is equivalent to step function.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action_id (int): the action predicted by the agent
        '''
        return self.step(agent_state)

    def step(self, agent_state) -> int:
        ''' Predict the action_id given the current state.
            Rookie01 strategy:
                Case where can gin:
                    Choose one of the gin actions.
                Case where can knock:
                    Choose one of the knock actions.
                Case where can discard:
                    Gin if can. Knock if can.
                    Otherwise, put aside cards in some best meld cluster.
                    Choose one of the remaining cards with highest deadwood value.
                    Discard that card.
                Case where can pick up discard:
                    Pick up discard card if it forms a worthwhile meld else draw a card (or declare dead hand if cannot draw).
                Case otherwise:
                    Choose a random action.

        Args:
            agent_state (numpy.array): an numpy array that represents the current state

        Returns:
            action_id (int): the action_id predicted
        '''
        agent_action_ids = agent_state['raw_agent_actions']
        action_ids = agent_action_ids.copy()
        legal_action_events = [ActionEvent.decode_action(x) for x in agent_action_ids]
        gin_action_events = [x for x in legal_action_events if isinstance(x, GinAction)]
        knock_action_events = [x for x in legal_action_events if isinstance(x, KnockAction)]
        discard_action_events = [x for x in legal_action_events if isinstance(x, DiscardAction)]
        env_hand = agent_state['obs'][0]
        hand = gin_rummy_utils.decode_cards(env_cards=env_hand)
        if gin_action_events:
            action_ids = [x.action_id for x in gin_action_events]
        elif knock_action_events:
            action_ids = [x.action_id for x in knock_action_events]
        elif discard_action_events:
            action_ids = self._decide_discard_action_ids(hand=hand, discard_action_events=discard_action_events)
        elif pick_up_discard_action_id in agent_action_ids:
            env_top_discard_card = agent_state['obs'][1]
            top_discard_card = gin_rummy_utils.decode_cards(env_cards=env_top_discard_card)[0]
            action_ids = self._decide_get_card_action_ids(top_discard_card=top_discard_card, hand=hand, agent_action_ids=agent_action_ids)
        return np.random.choice(action_ids)

    def _decide_discard_action_ids(self, hand: List[Card], discard_action_events: List[DiscardAction]) -> List[int]:
        candidate_discard_cards = [x.card for x in discard_action_events]
        best_discards = self._get_best_discards(candidate_discard_cards=candidate_discard_cards, hand=hand)
        if best_discards:
            action_ids = [DiscardAction(card=card).action_id for card in best_discards]
        else:
            action_ids = [x.action_id for x in discard_action_events]
        return action_ids

    @staticmethod
    def _decide_get_card_action_ids(top_discard_card: Card, hand: List[Card], agent_action_ids: List[int]) -> List[int]:
        thinker = Thinker(hand=hand)
        meld_piles_with_discard_card = thinker.get_meld_piles_with_discard_card(discard_card=top_discard_card)
        if meld_piles_with_discard_card:
            action_ids = [pick_up_discard_action_id]
        else:
            action_ids = [action for action in agent_action_ids]
            if len(action_ids) > 1:
                action_ids.remove(pick_up_discard_action_id)
        return action_ids

    @staticmethod
    def _get_best_discards(candidate_discard_cards: List[Card], hand: List[Card]) -> List[Card]:
        best_discards: List[Card] = []
        final_deadwood_count = 999
        for discard_card in candidate_discard_cards:
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