import panel as pn
import param

from World import World
from RLTrainerConfig import RLTrainerConfig

class RLTrainerPane(pn.pane.Markdown):

    def __init__(self, world: World):
        super().__init__()
        defaultConfig = RLTrainerConfig()
        rl_trainer_config = world.rl_trainer_config
        markdown = f"""
            <div class="special_table"></div>
            | Name | Value | Default |
            | :--: | :--: | :--: |
            | algorithm | {rl_trainer_config.algorithm} | {defaultConfig.algorithm} |
            | num_episodes | {rl_trainer_config.num_episodes} | {defaultConfig.num_episodes} |
            | num_eval_games | {rl_trainer_config.num_eval_games} | {defaultConfig.num_eval_games} |
            | evaluate_every | {rl_trainer_config.evaluate_every} | {defaultConfig.evaluate_every} |

            log_dir: {world.agent_dir}
            """
        self.sizing_mode = 'stretch_width'
        self.object = markdown

class RLTrainerConfigParam(param.Parameterized):

    max_num_episodes = 100000
    max_num_eval_games = 100
    max_evaluate_every = 10000

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

class RLTrainerConfigWindow:

    def __init__(self, world: World):
        self.world = world
        self.trainer_param = RLTrainerConfigParam(world=world)
        self.content_pane = self.get_content_pane()
        # watch
        self.trainer_param.param.watch(fn=self.update, parameter_names=list(self.trainer_param.param))
    
    @property
    def window(self):
        margin = [0, 0, 10, 30]
        widgets = {
            'algorithm': {'widget_type': pn.widgets.Select, 'margin': margin, 'max_width': 60},
            'num_episodes': {'widget_type': pn.widgets.IntInput, 'margin': margin, 'max_width': 100},
            'num_eval_games': {'widget_type': pn.widgets.IntInput, 'margin': margin, 'max_width': 100},
            'evaluate_every': {'widget_type': pn.widgets.IntInput, 'margin': margin, 'max_width': 100},
            }
        window_content = pn.Row(pn.Param(self.trainer_param.param, name="RL Trainer Config Update", widgets=widgets), self.content_pane)
        window_content.margin = [10, 0, 10, 10]
        window = pn.Row(window_content)
        window.background = 'green'
        return window

    def update(self, event):
        self.world.rl_trainer_config.algorithm = self.trainer_param.algorithm
        self.world.rl_trainer_config.num_episodes = self.trainer_param.num_episodes
        self.world.rl_trainer_config.num_eval_games = self.trainer_param.num_eval_games
        self.world.rl_trainer_config.evaluate_every = self.trainer_param.evaluate_every
        self.content_pane.objects = self.get_content_pane().objects
    
    def get_content_pane(self):
        title = pn.pane.Markdown(f'### RL Trainer Settings')
        trainer_pane = RLTrainerPane(world=self.world)
        content_pane = pn.Column(title, trainer_pane)
        content_pane.sizing_mode = 'stretch_width'
        return content_pane