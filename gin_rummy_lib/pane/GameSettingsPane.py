import panel as pn

from rlcard.games.gin_rummy.utils.settings import Setting

from World import World
from util import game_settings_to_dict

class GameSettingsPane(pn.Column):

    def __init__(self, world: World):
        super().__init__()

        ### title_pane
        title_pane = pn.pane.Markdown(f"# Game Settings")

        ### default_setting_pane
        setting = Setting.default_setting()
        lines = ["### Default"]
        for key, value in setting.items():
            lines.append(f'{value}')
            lines.append("\n")
        default_setting_pane = pn.pane.Markdown("\n".join(lines))

        ### setting_pane
        lines = ["### Current Game Settings"]
        game_settings_dict = game_settings_to_dict(settings=world.game.settings)
        if game_settings_dict:
            for key, value in game_settings_dict.items():
                lines.append(f'{key}: {value}')
                lines.append("\n")
        setting_pane = pn.pane.Markdown("\n".join(lines))

        self.append(title_pane)
        self.append(pn.Row(setting_pane, default_setting_pane))

        self.width = 1200
