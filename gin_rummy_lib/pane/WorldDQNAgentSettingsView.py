import os
import torch

import panel as pn
import param

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.agents import DQNAgent

from DQNAgentConfig import DQNAgentConfig
from World import World

from util import to_int_list

class WorldDQNAgentSettingsView(param.Parameterized):

    max_replay_memory_size = 20000
    max_replay_memory_init_size = 1000
    max_update_target_estimator_every = 1000
    max_discount_factor = 1.00
    max_epsilon_start = 1.0
    max_epsilon_end = 0.5 # 0.1
    max_epsilon_decay_steps = 20000
    max_batch_size = 128
    max_train_every = 10
    max_learning_rate = 1.0

    replay_memory_size = param.Integer(bounds=(0, max_replay_memory_size)) # 20000
    replay_memory_init_size = param.Integer(bounds=(0, max_replay_memory_init_size)) # 100
    update_target_estimator_every = param.Integer(bounds=(0, max_update_target_estimator_every))
    discount_factor = param.Number(bounds=(0, max_discount_factor)) # 0.99
    epsilon_start = param.Number(bounds=(0, max_epsilon_start)) # 1.0
    epsilon_end = param.Number(bounds=(0, max_epsilon_end)) # 0.01 # 0.1
    epsilon_decay_steps = param.Integer(bounds=(0, max_epsilon_decay_steps))
    batch_size = param.Integer(bounds=(0, max_batch_size)) # 32
    train_every = param.Integer(bounds=(0, max_train_every)) # 1
    learning_rate = param.Number(bounds=(0, max_learning_rate)) # 0.00005 #0.00005
    state_shape = param.String() # [5, 52]
    mlp_layers = param.String() # [128, 128, 128]  # [128, 128, 128] # [64, 64, 64] # [64, 64]
    model_name = param.String()

    create_dqn_agent_button = param.Action(lambda x: x.param.trigger('create_dqn_agent_button'), label='Create DQN agent')

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        dqn_agent_config = world.dqn_agent_config
        self.replay_memory_size = min(dqn_agent_config.replay_memory_size, self.max_replay_memory_size)
        self.replay_memory_init_size = min(dqn_agent_config.replay_memory_init_size, self.max_replay_memory_init_size)
        self.update_target_estimator_every = min(dqn_agent_config.update_target_estimator_every, self.max_update_target_estimator_every)
        self.discount_factor = min(dqn_agent_config.discount_factor, self.max_discount_factor)
        self.epsilon_start = min(dqn_agent_config.epsilon_start, self.max_epsilon_start)
        self.epsilon_end = min(dqn_agent_config.epsilon_end, self.max_epsilon_end)
        self.epsilon_decay_steps = min(dqn_agent_config.epsilon_decay_steps, self.max_epsilon_decay_steps)
        self.batch_size = min(dqn_agent_config.batch_size, self.max_batch_size)
        self.train_every = min(dqn_agent_config.train_every, self.max_train_every)
        self.learning_rate = min(dqn_agent_config.learning_rate, self.max_learning_rate)
        self.state_shape = str(dqn_agent_config.state_shape)
        self.mlp_layers = str(dqn_agent_config.mlp_layers)
        self.model_name = dqn_agent_config.model_name

    # method keeps on watching whether button is triggered
    @param.depends('create_dqn_agent_button', watch=True)
    def create_dqn_agent(self):
        agent = None
        #device = get_device() # Check whether gpu is available
        agent_path = self.world.agent_path
        if os.path.exists(agent_path):
            pass
        else:
            exception_error = None
            try: # Note this: kludge
                state_shape = to_int_list(self.state_shape)
            except Exception as error:
                exception_error = error
            try: # Note this: kludge
                mlp_layers = to_int_list(self.mlp_layers)
            except Exception as error:
                exception_error = error
            if not exception_error:
                game = GinRummyGame() # Note this
                num_actions = game.get_num_actions()
                agent = DQNAgent(
                    replay_memory_size=self.replay_memory_size,
                    replay_memory_init_size=self.replay_memory_init_size,
                    update_target_estimator_every=self.update_target_estimator_every,
                    discount_factor=self.discount_factor,
                    epsilon_start=self.epsilon_start,
                    epsilon_end=self.epsilon_end,
                    epsilon_decay_steps=self.epsilon_decay_steps,
                    batch_size=self.batch_size,
                    num_actions=num_actions, # Note this: kludge
                    state_shape=state_shape, # Note this: kludge
                    train_every=self.train_every,
                    mlp_layers=mlp_layers, # Note this: kludge
                    #device=device,
                    learning_rate=self.learning_rate)
                torch.save(agent, agent_path)
                self.batch_size = 60
                #print(f'train_steps={agent.train_t} time_steps={agent.total_t}')
                #print(agent.q_estimator.qnet)
        # FIXME: show fail/success message

    @property
    def view(self):
        pn.widgets.IntInput.margin = [0, 80]
        pn.widgets.FloatInput.margin = [0, 80]
        pn.widgets.FloatInput.step = 0.01
        widgets = {
            'replay_memory_size': pn.widgets.IntInput,
            'replay_memory_init_size': pn.widgets.IntInput,
            'update_target_estimator_every': pn.widgets.IntInput,
            'discount_factor': pn.widgets.FloatInput,
            'epsilon_start': pn.widgets.FloatInput,
            'epsilon_end': pn.widgets.FloatInput,
            'epsilon_decay_steps': pn.widgets.IntInput,
            'batch_size': pn.widgets.IntInput,
            'train_every': pn.widgets.IntInput,
            'learning_rate': pn.widgets.FloatInput
            }
        return pn.Row(pn.Param(self.param, widgets=widgets), self.content)

    def content(self):
        self.update()
        defaultConfig = DQNAgentConfig()
        config = self.world.dqn_agent_config
        body = pn.pane.Markdown(f"""
            ### DQN Agent Settings
            <div class="special_table"></div>
            | Name | Value | Default |
            | :--: | :--: | :--: |
            | replay_memory_size | {config.replay_memory_size} | {defaultConfig.replay_memory_size} |
            | replay_memory_init_size | {config.replay_memory_init_size} | {defaultConfig.replay_memory_init_size} |
            | update_target_estimator_every | {config.update_target_estimator_every} | {defaultConfig.update_target_estimator_every} |
            | discount_factor | {config.discount_factor} | {defaultConfig.discount_factor} |
            | epsilon_start | {config.epsilon_start} | {defaultConfig.epsilon_start} |
            | epsilon_end | {config.epsilon_end} | {defaultConfig.epsilon_end} |
            | epsilon_decay_steps | {config.epsilon_decay_steps} | {defaultConfig.epsilon_decay_steps} |
            | batch_size | {config.batch_size} | {defaultConfig.batch_size} |
            | train_every | {config.train_every} | {defaultConfig.train_every} |
            | learning_rate | {config.learning_rate} | {defaultConfig.learning_rate} |
            | state_shape | {config.state_shape} | {defaultConfig.state_shape} |
            | mlp_layers | {config.mlp_layers} | {defaultConfig.mlp_layers} |
            | model_name | {config.model_name} | {defaultConfig.model_name} |
            """)
        return pn.Column(body)
    
    def update(self):
        world = self.world
        dqn_agent_config = world.dqn_agent_config
        dqn_agent_config.replay_memory_size = self.replay_memory_size
        dqn_agent_config.replay_memory_init_size = self.replay_memory_init_size
        dqn_agent_config.update_target_estimator_every = self.update_target_estimator_every
        dqn_agent_config.discount_factor = self.discount_factor
        dqn_agent_config.epsilon_start = self.epsilon_start
        dqn_agent_config.epsilon_end = self.epsilon_end
        dqn_agent_config.epsilon_decay_steps = self.epsilon_decay_steps
        dqn_agent_config.batch_size = self.batch_size
        dqn_agent_config.train_every = self.train_every
        dqn_agent_config.learning_rate = self.learning_rate
        try:
            dqn_agent_config.state_shape = to_int_list(self.state_shape)
        except:
            pass
        try:
            dqn_agent_config.mlp_layers = to_int_list(self.mlp_layers)
        except:
            pass
        dqn_agent_config.model_name = self.model_name