import panel as pn
import param

from rlcard.games.gin_rummy.utils.settings import Setting

from World import World
from pane.GameSettingsPane import GameSettingsPane

class WorldGameSettingsView(param.Parameterized):

    dealer_for_round = param.Selector(objects=[0, 1, 2])
    stockpile_dead_card_count = param.Integer(bounds=(0, 52))
    going_out_deadwood_count = param.Integer(bounds=(0, 50))
    max_drawn_card_count = param.Integer(bounds=(0, 52))
    max_move_count = param.Integer(bounds=(0, 200))
    is_allowed_knock = param.Boolean(True)
    is_allowed_gin = param.Boolean(True)
    is_allowed_pick_up_discard = param.Boolean(True)
    is_allowed_to_discard_picked_up_card = param.Boolean(False)
    is_always_knock = param.Boolean(False)
    is_south_never_knocks = param.Boolean(False)

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        for key, value in Setting.default_setting().items():
            if key == Setting.dealer_for_round:
                self.dealer_for_round = world.game_settings.at[0, key]
            elif key == Setting.stockpile_dead_card_count:
                self.stockpile_dead_card_count = int(world.game_settings.at[0, key])
            elif key == Setting.going_out_deadwood_count:
                self.going_out_deadwood_count = int(world.game_settings.at[0, key])
            elif key == Setting.max_drawn_card_count:
                self.max_drawn_card_count = int(world.game_settings.at[0, key])
            elif key == Setting.max_move_count:
                self.max_move_count = int(world.game_settings.at[0, key])
            elif key == Setting.is_allowed_knock:
                self.is_allowed_knock = bool(world.game_settings.at[0, key])
            elif key == Setting.is_allowed_gin:
                self.is_allowed_gin = bool(world.game_settings.at[0, key])
            elif key == Setting.is_allowed_pick_up_discard:
                self.is_allowed_pick_up_discard = bool(world.game_settings.at[0, key])
            elif key == Setting.is_allowed_to_discard_picked_up_card:
                self.is_allowed_to_discard_picked_up_card = bool(world.game_settings.at[0, key])
            elif key == Setting.is_always_knock:
                self.is_always_knock = bool(world.game_settings.at[0, key])
            elif key == Setting.is_south_never_knocks:
                 self.is_south_never_knocks = bool(world.game_settings.at[0, key])
    @property
    def view(self):
        return pn.Row(self.param, self.content)

    def content(self):
        self.update()
        gameSettingsPane = GameSettingsPane(world=self.world)
        return pn.Column(gameSettingsPane)
    
    def update(self):
        world = self.world
        world.game_settings.at[0, Setting.dealer_for_round] = self.dealer_for_round
        world.game_settings.at[0, Setting.stockpile_dead_card_count] = self.stockpile_dead_card_count
        world.game_settings.at[0, Setting.going_out_deadwood_count] = self.going_out_deadwood_count
        world.game_settings.at[0, Setting.max_drawn_card_count] = self.max_drawn_card_count
        world.game_settings.at[0, Setting.max_move_count] = self.max_move_count
        world.game_settings.at[0, Setting.is_allowed_knock] = self.is_allowed_knock
        world.game_settings.at[0, Setting.is_allowed_gin] = self.is_allowed_gin
        world.game_settings.at[0, Setting.is_allowed_pick_up_discard] = self.is_allowed_pick_up_discard
        world.game_settings.at[0, Setting.is_allowed_to_discard_picked_up_card] = self.is_allowed_to_discard_picked_up_card
        world.game_settings.at[0, Setting.is_always_knock] = self.is_always_knock
        world.game_settings.at[0, Setting.is_south_never_knocks] = self.is_south_never_knocks
        world.save_game_settings() # Note: this will occur every time a value is changed