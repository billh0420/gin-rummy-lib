import panel as pn
import os

from rlcard.agents import DQNAgent

import panel as pn

from rlcard.agents import DQNAgent

from DQNAgentConfig import DQNAgentConfig

class DQNAgentPane(pn.pane.Markdown):

    def __init__(self, dqn_agent: DQNAgent):
        super().__init__()
        markdown = self.get_markdown(dqn_agent=dqn_agent)
        self.width_policy = 'max'
        self.object = markdown
    
    def get_markdown(self, dqn_agent: DQNAgent):
        defaultConfig = DQNAgentConfig()
        model_name = ''
        save_path = dqn_agent.save_path
        if save_path:
            model_name = os.path.basename(dqn_agent.save_path)
        markdown = f"""
            <div class="special_table"></div>
            | Name | Value | Default |
            | :--: | :--: | :--: |
            | replay_memory_size | {dqn_agent.memory.memory_size} | {defaultConfig.replay_memory_size} |
            | replay_memory_init_size | {dqn_agent.replay_memory_init_size} | {defaultConfig.replay_memory_init_size} |
            | update_target_estimator_every | {dqn_agent.update_target_estimator_every} | {defaultConfig.update_target_estimator_every} |
            | discount_factor | {dqn_agent.discount_factor} | {defaultConfig.discount_factor} |
            | epsilon_start | {dqn_agent.epsilons[0]} | {defaultConfig.epsilon_start} |
            | epsilon_end | {dqn_agent.epsilons[-1]} | {defaultConfig.epsilon_end} |
            | epsilon_decay_steps | {dqn_agent.epsilon_decay_steps} | {defaultConfig.epsilon_decay_steps} |
            | batch_size | {dqn_agent.batch_size} | {defaultConfig.batch_size} |
            | train_every | {dqn_agent.train_every} | {defaultConfig.train_every} |
            | save_every | {dqn_agent.save_every} | {defaultConfig.save_every} |
            | learning_rate | {dqn_agent.q_estimator.learning_rate} | {defaultConfig.learning_rate} |
            | num_actions | {dqn_agent.q_estimator.num_actions} | {defaultConfig.num_actions} |
            | state_shape | {dqn_agent.q_estimator.state_shape} | {defaultConfig.state_shape} |
            | mlp_layers | {dqn_agent.q_estimator.mlp_layers} | {defaultConfig.mlp_layers} |
            | model_name | {model_name} | {defaultConfig.model_name} |
        """
        return markdown