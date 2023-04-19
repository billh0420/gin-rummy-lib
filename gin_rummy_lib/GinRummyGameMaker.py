import os
import pandas as pd

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings

from GinRummyScorer230402 import GinRummyScorer230402
from GameMaker import GameMaker
from util import game_settings_to_dict

class GinRummyGameMaker(GameMaker):

    def __init__(self, selection: str):
        self.selection = selection

    def make_game(self):
        game = None
        if self.selection == 'win_or_lose':
            game = self.make_game_win_or_lose()
        return game

    def make_game_win_or_lose(self) -> GinRummyGame:
        game = GinRummyGame()
        game.settings.max_move_count = 80
        game.judge.scorer = GinRummyScorer230402()
        return game