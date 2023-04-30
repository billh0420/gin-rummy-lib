from __future__ import annotations

from rlcard.envs.gin_rummy import GinRummyEnv

from rlcard.games.gin_rummy.game import GinRummyGame

class Contest:

    def __init__(self, game: GinRummyGame, delegate: Contest_Delegate):
        self.game = game
        self.delegate = delegate
        config = {'allow_step_back': False, 'seed': None}
        env = GinRummyEnv(config=config)
        env.game = self.game
        self.env = env

    def compete(self, num_rounds: int, agent, opponents):
        self.delegate.will_start_competition(game=self.game, agent=agent)
        for opponent in opponents:
            self.delegate.will_start_match(game=self.game, agent=agent, opponent=opponent)
            for round in range(num_rounds):
                self.delegate.will_start_game(game=self.game, agent=agent, opponent=opponent)
                self.play_game(agent=agent, opponent=opponent)
                self.delegate.did_finish_game(game=self.game, agent=agent, opponent=opponent)
            self.delegate.did_finish_match(game=self.game, agent=agent, opponent=opponent)
        self.delegate.did_finish_competition(game=self.game, agent=agent)

    def play_game(self, agent, opponent):
        agents = [agent, opponent]
        state, player_id = self.game.init_game()
        while not self.game.is_over():
            player = self.game.round.players[player_id]
            agent = agents[player_id]
            env_state = self.env.get_state(player_id)
            best_action, info = agent.eval_step(env_state)
            game_action = self.game.decode_action(best_action)
            self.delegate.will_play_move(game=self.game, agent=agent, opponent=opponent, action=game_action)
            state, player_id = self.game.step(game_action)
            self.delegate.did_play_move(game=self.game, agent=agent, opponent=opponent, action=game_action)

class Contest_Delegate:

    def will_start_competition(self, game, agent):
        print(f'will_start_competition')

    def will_start_match(self, game, agent, opponent):
        pass

    def will_start_game(self, game, agent, opponent):
        pass

    def will_play_move(self, game, agent, opponent, action):
        pass

    def did_play_move(self, game, agent, opponent, action):
        pass

    def did_finish_game(self, game, agent, opponent):
        pass

    def did_finish_match(self, game, agent, opponent):
        pass

    def did_finish_competition(self, game, agent):
        print(f'did_finish_competition')