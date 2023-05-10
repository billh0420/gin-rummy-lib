# FIXME
#   Need to eliminate GinRummyEnv:
#       1) Can I drop state from GinRummy?

import pandas as pd

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.action_event import ActionEvent

from GameObserver import GameObserver

class GameMatch:

    def __init__(self, game:GinRummyGame, agents, max_review_episodes:int):
        self.max_review_episodes = max_review_episodes
        self.dataframe = self.get_dataframe(game=game, agents=agents)

    def get_dataframe(self, game:GinRummyGame, agents):
        game_observer = GameObserver()
        rows = []
        episode = 0
        while episode < self.max_review_episodes:
            episode += 1
            state, player_id = game.init_game()
            step = 0
            while game.is_over() == False:
                step += 1
                agent = agents[player_id]
                agent_state = agent.get_agent_state(player_id=player_id, game=game)
                best_action_id, info = agent.eval_step(state=agent_state)
                best_action = ActionEvent.decode_action(action_id=best_action_id)
                row = self.get_player_row(player_id, best_action_id, agent_state, game)
                rows.append(row)
                state, player_id = game.step(action=best_action)
                if step >= 200: # Note this
                    break
        dataframe = pd.DataFrame(rows)
        return dataframe

    def get_player_row(self, player_id: int, action_id: int, agent_state, game: GinRummyGame):
        opponent_id = (player_id + 1) % 2
        legal_actions = game.judge.get_legal_actions()
        legal_action_ids = list(map(lambda legal_action: legal_action.action_id, legal_actions))
        held_cards = game.round.players[player_id].hand.copy()
        opponent = game.round.players[opponent_id]
        opponent_known_cards = opponent.known_cards.copy()
        discard_pile = game.round.dealer.discard_pile
        unknown_cards = game.round.dealer.stock_pile + [card for card in opponent.hand if card not in opponent_known_cards]
        row = {
            'player_id': player_id,
            'held_cards': held_cards,
            'top_card': discard_pile[-1:].copy(),
            'dead_cards': discard_pile[:-1].copy(),
            'opponent_known_cards': opponent_known_cards,
            'unknown_cards': unknown_cards,
            'legal_action_ids': legal_action_ids,
            'action_id': action_id,
            'agent_state': agent_state}
        return row
