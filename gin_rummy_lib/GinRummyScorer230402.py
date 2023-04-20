# Gin Rummy Scoring function
#
# A player gets 1 point for going out (knock or gin) else 0 points (dead hand or lose).
#
# The DQNAgent needs to learn how to form and keep melds so that he can go out.
#
# The DQNAgent should pickup the discard when it would form a meld. Otherwise, he should draw a card (altough an exper player might break this rule sometimes).

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.player import GinRummyPlayer
from rlcard.games.gin_rummy.utils.action_event import KnockAction, GinAction
from rlcard.games.gin_rummy.utils.scorers import GinRummyScorer

class GinRummyScorer230402(GinRummyScorer):

    def __init__(self):
        self.name = 'GinRummyScorer230402: win_or_lose'

    def get_payoffs(self, game: GinRummyGame):
        payoffs = [0, 0]
        for i in range(2):
            player = game.round.players[i]
            payoff = self.get_payoff(player=player, game=game)
            payoffs[i] = payoff
        return payoffs

    def get_payoff(self, player: GinRummyPlayer, game: GinRummyGame) -> float:
        ''' Get the payoff of player:
                a) 1.0 if player gins
                b) 1.0 if player knocks
                c) 0.0 otherwise
            The goal is to have the agent learn how to knock and gin.
        Returns:
            payoff (int or float): payoff for player (higher is better)
        '''
        going_out_action = game.round.going_out_action
        going_out_player_id = game.round.going_out_player_id
        if going_out_player_id == player.player_id and isinstance(going_out_action, KnockAction):
            payoff = 1
        elif going_out_player_id == player.player_id and isinstance(going_out_action, GinAction):
            payoff = 1
        else:
            payoff = 0
        return payoff