import panel as pn
import torch
import os

from rlcard.agents import DQNAgent

from DQNAgent_230506 import DQNAgent_230506
from DQNAgentConfig import DQNAgentConfig
from util import to_int_list

class DQNAgentConfigWindow(pn.Column):

    @staticmethod
    def create_dqn_agent(config: DQNAgentConfig, world_dir:str) -> DQNAgent or None:
        agent = None
        agent_dir = f'{world_dir}/agents/{config.model_name}'
        agent_path = f'{agent_dir}/{config.model_name}.pth'
        if os.path.exists(agent_path):
            agent = torch.load(agent_path) # FIXME: 230421 is this ok?
        else:
            agent = DQNAgent_230506(config=config)
            if not os.path.exists(agent_dir):
                os.makedirs(agent_dir)
            torch.save(agent, agent_path)
        return agent

    def __init__(self, world_dir:str = '.'):
        super().__init__()
        self.world_dir = world_dir
        self.dqn_agent_config = DQNAgentConfig()

        window_title = pn.pane.Markdown("# DQN Agent Config Window")
        self.controls = DQNAgentConfigControls(dqn_agent_config=self.dqn_agent_config)

        title = pn.pane.Markdown("### DQN Agent Config Settings")
        self.dqn_agent_config_pane = DQNAgentConfigPane(dqn_agent_config=self.dqn_agent_config)
        self.dqn_agent_config_view = pn.Column(title, self.dqn_agent_config_pane)
        self.dqn_agent_config_view.width_policy = 'max'
        self.dqn_agent_config_view.height_policy = 'max'

        content = pn.Row(self.controls, self.dqn_agent_config_view)
        content.margin = [0, 10, 10, 10]
        self.append(window_title)
        self.append(content)
        self.width_policy = 'max'
        self.background = 'green'

        ### hook up value controls
        for control in self.controls.value_controls:
            control.param.watch(self.update, 'value')

        ### hook up create_dqn_agent_button
        self.controls.create_dqn_agent_button.on_click(self.on_click_create_dqn_agent)

    def on_click_create_dqn_agent(self, event):
        dqn_agent_config = self.dqn_agent_config
        dqn_agent = DQNAgentConfigWindow.create_dqn_agent(config=dqn_agent_config, world_dir=self.world_dir)
        # FIXME: show fail/success message
        if dqn_agent is None:
            print(f'Cannot create dqn_agent.')
        # agent = None
        # #device = get_device() # Check whether gpu is available
        # agent_path = self.world.agent_path
        # if os.path.exists(agent_path):
        #     pass
        # else:
        #     exception_error = None
        #     try: # Note this: kludge
        #         state_shape = to_int_list(self.state_shape)
        #     except Exception as error:
        #         exception_error = error
        #     try: # Note this: kludge
        #         mlp_layers = to_int_list(self.mlp_layers)
        #     except Exception as error:
        #         exception_error = error
        #     if not exception_error:
        #         game = GinRummyGame() # Note this
        #         num_actions = game.get_num_actions()
        #         agent = DQNAgent(
        #             replay_memory_size=self.replay_memory_size,
        #             replay_memory_init_size=self.replay_memory_init_size,
        #             update_target_estimator_every=self.update_target_estimator_every,
        #             discount_factor=self.discount_factor,
        #             epsilon_start=self.epsilon_start,
        #             epsilon_end=self.epsilon_end,
        #             epsilon_decay_steps=self.epsilon_decay_steps,
        #             batch_size=self.batch_size,
        #             num_actions=num_actions, # Note this: kludge
        #             state_shape=state_shape, # Note this: kludge
        #             train_every=self.train_every,
        #             save_every=self.save_every,
        #             mlp_layers=mlp_layers, # Note this: kludge
        #             #device=device,
        #             learning_rate=self.learning_rate)
        #         torch.save(agent, agent_path)
        #         self.batch_size = 60
        #         #print(f'train_steps={agent.train_t} time_steps={agent.total_t}')
        #         #print(agent.q_estimator.qnet)

    def update(self, event):
        dqn_agent_config = self.dqn_agent_config
        dqn_agent_config.replay_memory_size = self.controls.replay_memory_size_input.value
        dqn_agent_config.replay_memory_init_size = self.controls.replay_memory_init_size_input.value
        dqn_agent_config.update_target_estimator_every = self.controls.update_target_estimator_every_input.value
        dqn_agent_config.discount_factor = self.controls.discount_factor_input.value
        dqn_agent_config.epsilon_start = self.controls.epsilon_start_input.value
        dqn_agent_config.epsilon_end = self.controls.epsilon_end_input.value
        dqn_agent_config.epsilon_decay_steps = self.controls.epsilon_decay_steps_input.value
        dqn_agent_config.batch_size = self.controls.batch_size_input.value
        dqn_agent_config.train_every = self.controls.train_every_input.value
        dqn_agent_config.save_every = self.controls.save_every_input.value
        dqn_agent_config.learning_rate = self.controls.learning_rate_input.value
        dqn_agent_config.num_actions = self.controls.num_actions_input.value
        try:
            dqn_agent_config.state_shape = to_int_list(self.controls.state_shape_input.value)
        except:
            pass
        try:
            dqn_agent_config.mlp_layers = to_int_list(self.controls.mlp_layers_input.value)
        except:
            pass
        dqn_agent_config.model_name = self.controls.model_name_input.value

        self.dqn_agent_config_pane.object = self.dqn_agent_config_pane.get_markdown(dqn_agent_config=self.dqn_agent_config)

class DQNAgentConfigControls(pn.Row):

    def __init__(self, dqn_agent_config: DQNAgentConfig):
        super().__init__()

        # max values
        max_replay_memory_size = 20000
        max_replay_memory_init_size = 10000
        max_update_target_estimator_every = 10000
        max_discount_factor = 1.00
        max_epsilon_start = 1.0
        max_epsilon_end = 0.5 # 0.1
        max_epsilon_decay_steps = 20000
        max_batch_size = 128
        max_train_every = 1000
        max_save_every = 1000000
        max_learning_rate = 1.0
        max_num_actions = 1000

        # current values
        replay_memory_size = min(dqn_agent_config.replay_memory_size, max_replay_memory_size)
        replay_memory_init_size = min(dqn_agent_config.replay_memory_init_size, max_replay_memory_init_size)
        update_target_estimator_every = min(dqn_agent_config.update_target_estimator_every, max_update_target_estimator_every)
        discount_factor = min(dqn_agent_config.discount_factor, max_discount_factor)
        epsilon_start = min(dqn_agent_config.epsilon_start, max_epsilon_start)
        epsilon_end = min(dqn_agent_config.epsilon_end, max_epsilon_end)
        epsilon_decay_steps = min(dqn_agent_config.epsilon_decay_steps, max_epsilon_decay_steps)
        batch_size = min(dqn_agent_config.batch_size, max_batch_size)
        train_every = min(dqn_agent_config.train_every, max_train_every)
        save_every = min(dqn_agent_config.save_every, max_save_every)
        learning_rate = min(dqn_agent_config.learning_rate, max_learning_rate)
        state_shape = str(dqn_agent_config.state_shape)
        mlp_layers = str(dqn_agent_config.mlp_layers)
        model_name = dqn_agent_config.model_name

        num_actions = min(dqn_agent_config.num_actions, max_num_actions)

        # begin init
        self.margin_x = 10
        self.margin_y = 4

        self.value_controls = []
        self.columns = [pn.Column(), pn.Column()]

        self.column_index = 0
        self.replay_memory_size_input = self.make_int_input(name='replay_memory_size', value=replay_memory_size, start=1000, end=max_replay_memory_size, step=1000)
        self.replay_memory_init_size_input = self.make_int_input(name='replay_memory_init_size', value=replay_memory_init_size, start=1000, end=max_replay_memory_init_size, step=1000)
        self.update_target_estimator_every_input = self.make_int_input(name='update_target_estimator_every', value=update_target_estimator_every, start=1000, end=max_update_target_estimator_every, step=1000)
        self.discount_factor_input = self.make_float_input(name='discount_factor', value=discount_factor, end=max_discount_factor)
        self.epsilon_start_input = self.make_float_input(name='epsilon_start', value=epsilon_start, end=max_epsilon_start)
        self.epsilon_end_input = self.make_float_input(name='epsilon_end', value=epsilon_end, end=max_epsilon_end)
        self.epsilon_decay_steps_input = self.make_int_input(name='epsilon_decay_steps', value=epsilon_decay_steps, end=max_epsilon_decay_steps)
        self.batch_size_input = self.make_int_input(name='batch_size', value=batch_size, end=max_batch_size)
        self.train_every_input = self.make_int_input(name='train_every', value=train_every, end=max_train_every)
        self.save_every_input = self.make_int_input(name='save_every', value=save_every, end=max_save_every)
        self.learning_rate_input = self.make_float_input(name='learning_rate', value=learning_rate, end=max_learning_rate)

        self.column_index = 1
        self.num_actions_input = self.make_int_input(name='num_actions', value=num_actions, start=1, end=max_num_actions)
        self.state_shape_input = self.make_text_input(name='state_shape', value=state_shape)
        self.mlp_layers_input = self.make_text_input(name='mlp_layers', value=mlp_layers) # [128, 128, 128]  # [128, 128, 128] # [64, 64, 64] # [64, 64]

        self.model_name_input = self.make_text_input(name='model_name', value=model_name)
        self.create_dqn_agent_button = self.make_button(name='Create DQN agent')

        self.columns[0].margin = [0, 100, 0, 0] # kludge

        self.append(self.columns[0])
        self.append(self.columns[1])

    def make_int_input(self, name, value=0, start=0, end=10, step=1):
        result = pn.widgets.IntInput(name=name, value=value, start=start, end=end, step=step)
        result.min_width = 100
        result.max_width = result.min_width
        result.width_policy ='max'
        result.margin = [self.margin_y, self.margin_x, self.margin_y, self.margin_x]
        self.columns[self.column_index].append(result)
        self.value_controls.append(result)
        return result

    def make_float_input(self, name, value=0, start=0, end=1, step=0.01):
        result = pn.widgets.FloatInput(name=name, value=value, start=start, end=end, step=step)
        result.min_width = 100
        result.max_width = result.min_width
        result.width_policy ='min'
        result.margin = [self.margin_y, self.margin_x, self.margin_y, self.margin_x]
        self.columns[self.column_index].append(result)
        self.value_controls.append(result)
        return result

    def make_text_input(self, name: str, value: str = ''):
        result = pn.widgets.TextInput(name=name, value=value)
        self.columns[self.column_index].append(result)
        self.value_controls.append(result)
        return result

    def make_button(self, name: str):
        result = pn.widgets.Button(name=name)
        # result.width_policy = 'min'
        self.columns[self.column_index].append(result)
        return result

class DQNAgentConfigPane(pn.pane.Markdown):

    def __init__(self, dqn_agent_config:DQNAgentConfig):
        super().__init__()
        markdown = self.get_markdown(dqn_agent_config=dqn_agent_config)
        self.width_policy = 'max'
        self.object = markdown

    def get_markdown(self, dqn_agent_config: DQNAgentConfig):
        defaultConfig = DQNAgentConfig()
        markdown = f"""
            <div class="special_table"></div>
            | Name | Value | Default |
            | :--: | :--: | :--: |
            | replay_memory_size | {dqn_agent_config.replay_memory_size} | {defaultConfig.replay_memory_size} |
            | replay_memory_init_size | {dqn_agent_config.replay_memory_init_size} | {defaultConfig.replay_memory_init_size} |
            | update_target_estimator_every | {dqn_agent_config.update_target_estimator_every} | {defaultConfig.update_target_estimator_every} |
            | discount_factor | {dqn_agent_config.discount_factor} | {defaultConfig.discount_factor} |
            | epsilon_start | {dqn_agent_config.epsilon_start} | {defaultConfig.epsilon_start} |
            | epsilon_end | {dqn_agent_config.epsilon_end} | {defaultConfig.epsilon_end} |
            | epsilon_decay_steps | {dqn_agent_config.epsilon_decay_steps} | {defaultConfig.epsilon_decay_steps} |
            | batch_size | {dqn_agent_config.batch_size} | {defaultConfig.batch_size} |
            | train_every | {dqn_agent_config.train_every} | {defaultConfig.train_every} |
            | save_every | {dqn_agent_config.save_every} | {defaultConfig.save_every} |
            | learning_rate | {dqn_agent_config.learning_rate} | {defaultConfig.learning_rate} |
            | num_actions | {dqn_agent_config.num_actions} | {defaultConfig.num_actions} |
            | state_shape | {dqn_agent_config.state_shape} | {defaultConfig.state_shape} |
            | mlp_layers | {dqn_agent_config.mlp_layers} | {defaultConfig.mlp_layers} |
            | model_name | {dqn_agent_config.model_name} | {defaultConfig.model_name} |
        """
        return markdown
