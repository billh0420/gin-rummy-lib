import os
import pandas as pd

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings

from GinRummyScorer230402 import GinRummyScorer230402
from GameMaker import GameMaker
from util import game_settings_to_dict

class GinRummyGameMaker(GameMaker):

    def make_game(self):
        return None

class GameMakerWinOrLose02(GameMaker):

    def make_game(self):
        game = GinRummyGame()
        game.settings.max_move_count = 50
        game.judge.scorer = GinRummyScorer230402()
        return game