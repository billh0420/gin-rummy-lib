import panel as pn

from rlcard.games.gin_rummy.game import GinRummyGame

from GameMatch import GameMatch

class GameReviewer:

    def __init__(self, game:GinRummyGame, agents, view_width):
        self.game_match = GameMatch(game=game, agents=agents, max_review_episodes=0)
        self.select_using_filter_by_groups = self.make_select_using_filter_by_groups()
    
    def play_review_match(self, game:GinRummyGame, agents, max_review_episodes, view_width):
        self.game_match = GameMatch(game=game, agents=agents, max_review_episodes=max_review_episodes)
        self.select_using_filter_by_groups = self.make_select_using_filter_by_groups()

    @property
    def filtered_dataframe(self): # FIXME: 230402
        tau = self.select_using_filter_by_groups
        #print(f'tau={tau.value}')
        #print(f'tau.value={tau.value}')
        dataframe = self.game_match.dataframe
        return dataframe

    def make_select_using_filter_by_groups(self):
        filter_by_groups = {
            'Default': [
                "None"
            ],
            'Draw': [
                "Draw from discard pile does not make or extend a meld",
                "Draw from stock pile when draw from discard pile would make or extend a meld",
            ],
            'Discard': [
                "Discard card from a meld",
                "Discard card from a good pair",
                "Discard card from a poor pair",
            ],
            'Going out': [
                "Fail to knock when can",
                "Fail to gin when can",
            ]
        }
        select_using_filter_by_groups = pn.widgets.Select(name="Filter by", groups=filter_by_groups, width=550)
        return select_using_filter_by_groups