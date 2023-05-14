from collections import OrderedDict

from rlcard.games.gin_rummy.game import GinRummyGame

class GinRummyAgentActionsMixin:

    def get_agent_actions(self, player_id: int, game: GinRummyGame):
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("GinRummyAgentActionsMixin get_legal_actions: agent is not current player.")
        legal_actions = game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)
