from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.scorers import GinRummyScorer
from rlcard.games.gin_rummy.utils.move import KnockMove, GinMove, ScoreSouthMove
from rlcard.games.gin_rummy.utils import melding

class GinRummyScorer_WinOrLose(GinRummyScorer):

    def __init__(self):
        # super().__init__() # wch: super kludge 230502
        self.name = 'GinRummyScorer_WinOrLose: win_or_lose'

    def get_payoffs(self, game: GinRummyGame):
        payoffs = [0, 0]
        payoffs = [self.get_payoff(player=x, game=game) for x in game.round.players]
        return payoffs

    def get_payoff(self, player, game: GinRummyGame) -> float:
        ''' Get the payoff for player: 1 point for going out else 0
        Returns:
            payoff (int or float): payoff for player (higher is better)
        '''
        payoff = 0
        move_sheet = game.round.move_sheet
        if len(move_sheet) >= 3 and isinstance(move_sheet[-1], ScoreSouthMove):
            going_out_move = move_sheet[-3]
            if isinstance(going_out_move, KnockMove) or isinstance(going_out_move, GinMove):
                if going_out_move.player == player:
                    payoff = 1
        return payoff

class GinRummyScorer_MeldCredit(GinRummyScorer):

    def __init__(self):
        # super().__init__() # wch: super kludge 230502
        self.name = 'GinRummyScorer_MeldCredit: credit for melds'

    def get_payoffs(self, game: GinRummyGame):
        payoffs = [0, 0]
        payoffs = [self.get_payoff(player=x, game=game) for x in game.round.players]
        return payoffs

    def get_payoff(self, player, game: GinRummyGame) -> float:
        ''' Get the payoff for player: 1 point for each meld
        Returns:
            payoff (int or float): payoff for player (higher is better)
        '''
        credit = 0
        hand = player.hand
        meld_clusters = melding.get_meld_clusters(hand=hand)
        for meld_cluster in meld_clusters:
            meld_count = len(meld_cluster)
            meld_credit = 3 if meld_count == 3 else 2 if meld_count == 2 else 1 if meld_count == 1 else 0
            if meld_credit > credit:
                credit = meld_credit
        payoff = credit
        return payoff
