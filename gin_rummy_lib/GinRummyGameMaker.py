import os
import pandas as pd

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils.settings import Setting, Settings

from GinRummyScorer230402 import GinRummyScorer230402
from GameMaker import GameMaker

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
        settings = Settings()
        settings.change_settings(self.game_settings_dict)
        game.settings = settings
        game.judge.scorer = GinRummyScorer230402()
        return game

    @property
    def default_settings(self):
        if not os.path.exists(self.game_settings_path):
            default_game_settings = Setting.default_setting()
            default_game_settings_dict = dict((k.value, v) for (k, v) in default_game_settings.items())
            game_settings_df = pd.DataFrame([default_game_settings_dict])
            game_settings_df.to_json(self.game_settings_path)
        game_settings = pd.read_json(self.game_settings_path)
        return game_settings

    @property
    def game_settings_path(self):
        dir = os.path.abspath('.')
        return os.path.join(dir, f'game_settings.json')

    @property
    def game_settings_dict(self):
        result = dict()
        game_settings_dicts = self.default_settings.to_dict(orient='records')
        if len(game_settings_dicts) == 1:
            result = game_settings_dicts[0]
        return result