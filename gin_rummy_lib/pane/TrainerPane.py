import panel as pn

from World import World
from RLTrainerConfig import RLTrainerConfig

class TrainerPane(pn.pane.Markdown):

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