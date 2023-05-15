import numpy as np

from rlcard.games.gin_rummy.utils.action_event import ActionEvent, DiscardAction, KnockAction, GinAction
from rlcard.games.gin_rummy.utils.action_event import draw_card_action_id, declare_dead_hand_action_id
import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils

from GinRummyAgentStateMixin import GinRummyAgentStateMixin
from GinRummyAgentActionsMixin import GinRummyAgentActionsMixin
from GameAgent import GameAgent

class GinRummyLoserRuleAgent(GinRummyAgentStateMixin, GinRummyAgentActionsMixin, GameAgent):
    """
        Always gin if can.
        Always declare dead hand if can.
        Always draw a card if can.
        Always discards random card if can.
        Never knock if can.
        Never pick up a card if can.
    """

    def __init__(self):
        super().__init__()
        self.use_raw = False

    def eval_step(self, agent_state) -> int:
        ''' Predict the action_id given the current state.
            Loser strategy:
                Case where can gin:
                    Choose one of the gin actions.
                Case where can knock:
                    Never knock.
                Case where can discard:
                    Gin if can.
                    Done't knock if can.
                    Discard a random card.
                Case where can pick up discard:
                    Never pick up a card.
                Case otherwise:
                    Choose a random action.

        Args:
            agent_state (numpy.array): an numpy array that represents the current state

        Returns:
            action_id (int): the action_id predicted
        '''
        agent_action_ids = agent_state['raw_agent_actions']
        legal_action_events = [ActionEvent.decode_action(x) for x in agent_action_ids]
        gin_action_events = [x for x in legal_action_events if isinstance(x, GinAction)]
        knock_action_events = [x for x in legal_action_events if isinstance(x, KnockAction)]
        discard_action_events = [x for x in legal_action_events if isinstance(x, DiscardAction)]
        action_ids = []
        if gin_action_events:
            action_ids = [x.action_id for x in gin_action_events]
        elif declare_dead_hand_action_id in agent_action_ids:
            action_ids = [declare_dead_hand_action_id]
        elif draw_card_action_id in agent_action_ids:
            action_ids = [draw_card_action_id]
        elif discard_action_events:
            action_ids = [x.action_id for x in discard_action_events]
        else:
            action_ids = agent_action_ids
        action_id = np.random.choice(action_ids)
        return action_id
