import os

class DQNAgentConfig:

    def __init__(self):
        self.replay_memory_size = 200000 # 20000
        self.replay_memory_init_size = 1000 #100
        self.update_target_estimator_every = 1000
        self.discount_factor = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01 # 0.1
        self.epsilon_decay_steps = 20000
        self.batch_size = 128 #32
        self.train_every = 5 #1
        self.save_every = 1000000
        self.learning_rate=0.00005 #0.00005
        self.num_actions = 110
        self.state_shape = [5, 52]
        self.mlp_layers = [128, 128, 128]  # [128, 128, 128] # [64, 64, 64] # [64, 64]
        self.model_name ='dqn_agent'
        self.save_path = None
        self.use_raw = False
        self.device = None

    def to_dict(self):
        result = dict()
        result['state_shape'] = self.state_shape
        result['replay_memory_size'] = self.replay_memory_size
        result['replay_memory_init_size'] = self.replay_memory_init_size
        result['update_target_estimator_every'] = self.update_target_estimator_every
        result['discount_factor'] = self.discount_factor
        result['epsilon_start'] = self.epsilon_start
        result['epsilon_end'] = self.epsilon_end
        result['epsilon_decay_steps'] = self.epsilon_decay_steps
        result['batch_size'] = self.batch_size
        result['train_every'] = self.train_every
        result['save_every'] = self.save_every
        result['num_actions'] = self.num_actions
        result['mlp_layers'] = self.mlp_layers
        result['learning_rate'] = self.learning_rate
        result['model_name'] = self.model_name
        result['save_path'] = self.save_path
        result['use_raw'] = self.use_raw
        result['device'] = self.device
        return result

    def __str__(self):
        lines = []
        lines.append(f'----- DQNAgentConfig -----')
        lines.append(f'state_shape={self.state_shape}')
        lines.append(f'replay_memory_size={self.replay_memory_size}')
        lines.append(f'replay_memory_init_size={self.replay_memory_init_size}')
        lines.append(f'update_target_estimator_every={self.update_target_estimator_every}')
        lines.append(f'discount_factor={self.discount_factor}')
        lines.append(f'epsilon_start={self.epsilon_start}')
        lines.append(f'epsilon_end={self.epsilon_end}')
        lines.append(f'epsilon_decay_steps={self.epsilon_decay_steps}')
        lines.append(f'batch_size={self.batch_size}')
        lines.append(f'train_every={self.train_every}')
        lines.append(f'save_every={self.save_every}')
        lines.append(f'num_actions={self.num_actions}')
        lines.append(f'mlp_layers={self.mlp_layers}')
        lines.append(f'learning_rate={self.learning_rate}')
        lines.append(f'model_name={self.model_name}')
        lines.append(f'save_path={self.save_path}')
        lines.append(f'use_raw={self.use_raw}')
        lines.append(f'device={self.device}')
        return "\n".join(lines)
