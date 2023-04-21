import os
import pandas as pd

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings

from GinRummyScorer230402 import GinRummyScorer230402
from GameMaker import GameMaker
from util import game_settings_to_dict

class GinRummyGameMaker(GameMaker):

    def make_game(self):
        game = None
        config = self.get_game_maker_config()
        selection = config.get('selection')
        if selection == 'win_or_lose':
            game = self.make_game_win_or_lose(config=config)
        return game
    
    def get_game_maker_config(self):
        result = None
        file_path = 'game_maker_config.json' # cannot change this
        if os.path.exists(file_path):
            result = pd.read_json(path_or_buf=file_path, typ='series', orient='index')
        return result

    def make_game_win_or_lose(self, config) -> GinRummyGame:
        game = GinRummyGame()
        game.settings.max_move_count = config.get('max_move_count', default=60)
        game.judge.scorer = GinRummyScorer230402()
        return game