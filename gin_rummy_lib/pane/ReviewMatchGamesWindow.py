import panel as pn

from World import World
from pane.ReviewPlayWindow import ReviewPlayWindow
from pane.DQNAgentPane import DQNAgentPane
from pane.TrainerPane import TrainerPane
from pane.GameSettingsPane import GameSettingsPane

class ReviewMatchGamesWindow(pn.Tabs):

    def __init__(self, world: World):
        super().__init__()
        self.append(('Review', ReviewPlayWindow(world=world)))
        self.append(('DQNAgent', self.dqn_agent_view(world=world)))
        self.append(('Trainer', self.trainer_view(world=world)))
        self.append(('GameSettings', GameSettingsPane(world=world)))
        self.dynamic = True
        self.height = 800

    def dqn_agent_view(self, world):
        title = pn.pane.Markdown("# DQN Agent Settings")
        dqn_agent_pane = DQNAgentPane(dqn_agent=world.agent)
        dqn_agent_view = pn.Column(title, dqn_agent_pane)
        dqn_agent_view.width_policy = 'max'
        return dqn_agent_view

    def trainer_view(self, world):
        trainer_title = pn.pane.Markdown("# RL Trainer Settings")
        trainer_pane = TrainerPane(world=world)
        trainer_view = pn.Column(trainer_title, trainer_pane, sizing_mode = 'stretch_width')
        return trainer_view
