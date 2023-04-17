import panel as pn
import param

from World import World
from RLTrainerConfig import RLTrainerConfig
from pane.TrainerPane import TrainerPane

class WorldRLTrainerSettingsView(param.Parameterized):

    max_num_episodes = 20000
    max_num_eval_games = 100
    max_evaluate_every = 1000

    algorithm = param.Selector(objects=['dqn', 'nfsp'])
    num_episodes = param.Integer(bounds=(0, max_num_episodes))
    num_eval_games = param.Integer(bounds=(0, max_num_eval_games))
    evaluate_every = param.Integer(bounds=(0, max_evaluate_every)) #max(1, min(self.num_episodes // 20, 10000))

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.algorithm = world.rl_trainer_config.algorithm
        self.num_episodes = min(world.rl_trainer_config.num_episodes, self.max_num_episodes)
        self.num_eval_games = min(world.rl_trainer_config.num_eval_games, self.max_num_eval_games)
        self.evaluate_every = min(world.rl_trainer_config.evaluate_every, self.max_evaluate_every)

    @property
    def view(self):
        return pn.Row(self.param, self.content)

    def content(self):
        self.update()
        defaultConfig = RLTrainerConfig()
        config = self.world.rl_trainer_config
        title = pn.pane.Markdown(f'### RL Trainer Settings')
        body = TrainerPane(world=self.world)
        return pn.Column(title, body, sizing_mode = 'stretch_width')
    
    def update(self):
        world = self.world
        rl_trainer_config = world.rl_trainer_config
        rl_trainer_config.algorithm = self.algorithm
        rl_trainer_config.num_episodes = self.num_episodes
        rl_trainer_config.num_eval_games = self.num_eval_games
        rl_trainer_config.evaluate_every = self.evaluate_every
