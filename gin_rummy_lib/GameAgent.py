class GameAgent:

    def get_agent_state(self, player_id: int, game):
        raise NotImplementedError

    # def get_agent_action_ids(self, player_id: int, game: GinRummyGame):
    def get_agent_actions(self, player_id: int, game):
        raise NotImplementedError

    # def get_exploit_action_id(self, agent_state) -> int:
    def eval_step(self, agent_state) -> int:
        raise NotImplementedError
