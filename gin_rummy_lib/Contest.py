from __future__ import annotations

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.action_event import DrawCardAction
from rlcard.games.gin_rummy.utils.move import DeclareDeadHandMove

class Contest:

    def __init__(self, game: GinRummyGame, delegate: Contest_Delegate):
        self.game = game
        self.delegate = delegate

    def compete(self, num_rounds: int, agent, opponent):
        self.delegate.will_start_competition(game=self.game, agent=agent, opponent=opponent, num_rounds=num_rounds)
        for round in range(num_rounds):
            self.compete_game(agent=agent, opponent=opponent)
        self.delegate.did_finish_competition(game=self.game, agent=agent, opponent=opponent, num_rounds=num_rounds)

    def compete_game(self, agent, opponent):
        self.delegate.will_start_game(game=self.game, agent=agent, opponent=opponent)
        agents = [agent, opponent]
        state, player_id = self.game.init_game()
        tried_illegal_action = False
        move_at = 0
        while not self.game.is_over() and not tried_illegal_action:
            move_at += 1
            player = self.game.round.players[player_id]
            agent = agents[player_id]
            env_state = agent.get_env_state(player_id=player_id, game=self.game)
            best_action, info = agent.eval_step(env_state)
            if best_action not in [x.action_id for x in self.game.judge.get_legal_actions()]:
                tried_illegal_action = True
            else:
                game_action = self.game.decode_action(best_action)
                self.delegate.will_play_move(game=self.game, agent=agent, opponent=opponent, action=game_action, move_at=move_at)
                state, player_id = self.game.step(game_action)
                self.delegate.did_play_move(game=self.game, agent=agent, opponent=opponent, action=game_action, move_at=move_at)
        self.delegate.did_finish_game(game=self.game, agent=agent, opponent=opponent)

    @staticmethod
    def play_game(game: GinRummyGame, agent, opponent):
        class Delegate(Contest_Delegate):
            def will_play_move(self, game, agent, opponent, action, move_at: int):
                player = game.get_current_player()
                if isinstance(action, DrawCardAction):
                    card = game.round.dealer.stock_pile[-1]
                    print(f'{move_at:3}. {player} {action} {card}')
                else:
                    print(f'{move_at:3}. {player} {action}')
            def did_finish_game(self, game, agent, opponent):
                payoffs = game.judge.scorer.get_payoffs(game=game)
                print(f'payoffs={payoffs}')
        contest = Contest(game=game, delegate=Delegate())
        contest.compete_game(agent=agent, opponent=opponent)

    @staticmethod
    def play_match(num_rounds: int, game: GinRummyGame, agent, opponent):
        class Delegate(Contest_Delegate):
            def reset(self):
                self.total_payoffs = [0, 0]
                self.dead_hand_count = 0
            def will_start_competition(self, game, agent, opponent, num_rounds: int):
                self.reset()
            def did_finish_game(self, game, agent, opponent):
                payoffs = game.judge.scorer.get_payoffs(game=game)
                for player in game.round.players:
                    self.total_payoffs[player.player_id] += payoffs[player.player_id]
                if isinstance(game.round.move_sheet[-3], DeclareDeadHandMove):
                    self.dead_hand_count += 1
            def did_finish_competition(self, game, agent, opponent, num_rounds: int):
                print(f'average payoffs={[round(x / num_rounds, 4) for x in self.total_payoffs]}; dead_hand_count={self.dead_hand_count}')
        contest = Contest(game=game, delegate=Delegate())
        contest.compete(num_rounds=num_rounds, agent=agent, opponent=opponent)

class Contest_Delegate:

    def will_start_competition(self, game, agent, opponent, num_rounds: int):
        pass

    def will_start_game(self, game, agent, opponent):
        pass

    def will_play_move(self, game, agent, opponent, action, move_at: int):
        pass

    def did_play_move(self, game, agent, opponent, action, move_at: int):
        pass

    def did_finish_game(self, game, agent, opponent):
        pass

    def did_finish_competition(self, game, agent, opponent, num_rounds: int):
        pass
