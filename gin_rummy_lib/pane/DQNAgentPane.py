import panel as pn

from rlcard.agents import DQNAgent

class DQNAgentPane(pn.pane.Markdown):

    def __init__(self, dqn_agent: DQNAgent):
        super().__init__()
        #agent = world.agent
        markdown = self.get_markdown(dqn_agent=dqn_agent)
        self.width_policy = 'max'
        self.object = markdown
    
    def get_markdown(self, dqn_agent: DQNAgent):
        markdown = f"""
            <div class="special_table"></div>
            | Name | Value |
            | :--: | :--: |
            | replay_memory_size | {dqn_agent.memory.memory_size} |
            | replay_memory_init_size | {dqn_agent.replay_memory_init_size} |
            | update_target_estimator_every | {dqn_agent.update_target_estimator_every} |
            | discount_factor | {dqn_agent.discount_factor} |
            | epsilon_start | {dqn_agent.epsilons[0]} |
            | epsilon_end | {dqn_agent.epsilons[-1]} |
            | epsilon_decay_steps | {dqn_agent.epsilon_decay_steps} |
            | batch_size | {dqn_agent.batch_size} |
            | train_every | {dqn_agent.train_every} |
            | learning_rate | {dqn_agent.q_estimator.learning_rate} |
            | num_actions | {dqn_agent.q_estimator.num_actions} |
            | state_shape | {dqn_agent.q_estimator.state_shape} |
            | mlp_layers | {dqn_agent.q_estimator.mlp_layers} |
        """
        return markdown