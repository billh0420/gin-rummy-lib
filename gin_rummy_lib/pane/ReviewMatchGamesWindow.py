import panel as pn

from World import World
from pane.review_play_window.ReviewPlayWindow import ReviewPlayWindow
from pane.DQNAgentPane import DQNAgentPane
from pane.RLTrainerPane import RLTrainerPane
from pane.GameSettingsPane import GameSettingsPane

class ReviewMatchGamesWindow(pn.Tabs):

    def __init__(self, world: World):
        super().__init__()
        review_play_window = ReviewPlayWindow(world=world)
        review_play_window.background = 'green'
        self.append(('Review', review_play_window))
        self.append(('DQNAgent', self.dqn_agent_view(world=world)))
        self.append(('RL Trainer', self.rl_trainer_view(world=world)))
        self.append(('GameSettings', GameSettingsPane(world=world)))

    def dqn_agent_view(self, world):
        title = pn.pane.Markdown("# DQN Agent Settings")
        dqn_agent_pane = DQNAgentPane(dqn_agent=world.agent)
        dqn_agent_view = pn.Column(title, dqn_agent_pane)
        dqn_agent_view.width_policy = 'max'
        return dqn_agent_view

    def rl_trainer_view(self, world):
        rl_trainer_title = pn.pane.Markdown("# RL Trainer Settings")
        rl_trainer_pane = RLTrainerPane(world=world)
        rl_trainer_view = pn.Column(rl_trainer_title, rl_trainer_pane, sizing_mode = 'stretch_width')
        return rl_trainer_view