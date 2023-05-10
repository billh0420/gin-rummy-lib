''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from collections import OrderedDict

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils

from DQNAgentConfig import DQNAgentConfig
from GinRummyAgent import GinRummyAgent

from .Estimator import Estimator
from .Memory import Memory
from .Transition import Transition


class DQNAgent_230510(GinRummyAgent):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self, config: DQNAgentConfig):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
            save_path (str): The path to save the model checkpoints
            save_every (int): Save the model every X training steps
        '''

        replay_memory_size = 20000
        replay_memory_init_size = 100
        update_target_estimator_every = 1000
        discount_factor = 0.99
        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay_steps = 20000
        batch_size = 32
        num_actions = 2
        state_shape = None
        train_every = 1
        mlp_layers = None
        learning_rate = 0.00005
        device = None
        save_every = float('inf')
        model_name = 'dqn_agent'
        save_path = None
        use_raw = False
        device = None
        for key, value in config.to_dict().items():
            if key == 'replay_memory_size':
                replay_memory_size = value
            elif key == 'replay_memory_init_size':
                replay_memory_init_size = value
            elif key == 'update_target_estimator_every':
                update_target_estimator_every = value
            elif key == 'discount_factor':
                discount_factor = value
            elif key == 'epsilon_start':
                epsilon_start = value
            elif key == 'epsilon_end':
                epsilon_end = value
            elif key == 'epsilon_decay_steps':
                epsilon_decay_steps = value
            elif key == 'batch_size':
                batch_size = value
            elif key == 'num_actions':
                num_actions = value
            elif key == 'state_shape':
                state_shape = value
            elif key == 'train_every':
                train_every = value
            elif key == 'mlp_layers':
                mlp_layers = value
            elif key == 'learning_rate':
                learning_rate = value
            elif key == 'device':
                device = value
            elif key == 'save_every':
                save_every = value
            elif key == 'model_name':
                model_name = value
            elif key == 'save_path':
                save_path = value
            elif key == 'use_raw':
                use_raw = value
            elif key == 'device':
                device = value

        self.use_raw = use_raw
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every
        self.model_name = model_name

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def __repr__(self): # 230506
        lines = []
        lines.append(f'state_shape={self.q_estimator.state_shape}')
        lines.append(f'replay_memory_size={self.memory.memory_size}')
        lines.append(f'replay_memory_init_size={self.replay_memory_init_size}')
        lines.append(f'update_target_estimator_every={self.update_target_estimator_every}')
        lines.append(f'discount_factor={self.discount_factor}')
        lines.append(f'epsilon_start={self.epsilons[0]}')
        lines.append(f'epsilon_end={self.epsilons[-1]}')
        lines.append(f'epsilon_decay_steps={self.epsilon_decay_steps}')
        lines.append(f'batch_size={self.batch_size}')
        lines.append(f'train_every={self.train_every}')
        lines.append(f'save_every={self.save_every}')
        lines.append(f'num_actions={self.num_actions}')
        lines.append(f'mlp_layers={self.q_estimator.mlp_layers}')
        lines.append(f'learning_rate={self.q_estimator.learning_rate}')
        lines.append(f'model_name={self.model_name}')
        lines.append(f'save_path={self.save_path}')
        lines.append(f'device={self.device}')
        lines.append(f'use_raw={self.use_raw}')
        lines.append(f'Total timesteps: total_t={self.total_t}')
        lines.append(f'Total training steps: train_t={self.train_t}')
        return '\n'.join(lines)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device

    def get_agent_actions(self, player_id: int, game: GinRummyGame): # 230506
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("DQNAgent_230506 get_legal_actions: agent is not current player.")
        legal_actions = game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)

    def get_agent_state(self, player_id: int, game: GinRummyGame): # 230506
        if not game.is_over() and player_id != game.get_player_id():
            raise Exception("GinRummyKludge get_agent_state: agent is not current player.")
        agent_actions = self.get_agent_actions(player_id=player_id, game=game)
        player = game.round.players[player_id]
        opponent = game.round.players[(player_id + 1) % 2]
        stock_pile = game.round.dealer.stock_pile
        discard_pile = game.round.dealer.discard_pile
        top_discard = [] if not discard_pile else [discard_pile[-1]]
        if game.is_over():
            dead_cards = discard_pile[:-1]
            known_cards = opponent.hand
            unknown_cards = stock_pile
        else:
            dead_cards = discard_pile[:-1]
            known_cards = opponent.known_cards
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
        hand_rep = utils.encode_cards(player.hand)
        top_discard_rep = utils.encode_cards(top_discard)
        dead_cards_rep = utils.encode_cards(dead_cards)
        known_cards_rep = utils.encode_cards(known_cards)
        unknown_cards_rep = utils.encode_cards(unknown_cards)
        rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
        obs = np.array(rep)
        env_state = dict()
        env_state['obs'] = obs
        env_state['agent_actions'] = agent_actions
        env_state['raw_agent_actions'] = list(agent_actions.keys())
        env_state['raw_obs'] = obs
        return env_state

    def step(self, agent_state) -> int:
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            agent_state (numpy.array): current state

        Returns:
            action_id (int): an action id
        '''
        q_values = self.predict(agent_state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        agent_actions = list(agent_state['agent_actions'].keys())
        probs = np.ones(len(agent_actions), dtype=float) * epsilon / len(agent_actions)
        best_action_idx = agent_actions.index(np.argmax(q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)
        return agent_actions[action_idx]

    def eval_step(self, agent_state) -> int:
        ''' Predict the action for evaluation purpose.

        Args:
            agent_state (numpy.array): current state

        Returns:
            action_id (int): an action id
        '''
        q_values = self.predict(agent_state)
        best_action = np.argmax(q_values)
        return best_action

    def predict(self, agent_state):
        ''' Predict the masked Q-values

        Args:
            agent_state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        q_values = self.q_estimator.predict_nograd(np.expand_dims(agent_state['obs'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        agent_actions = list(agent_state['agent_actions'].keys())
        masked_q_values[agent_actions] = q_values[agent_actions]
        return masked_q_values

    def get_q_values_by_action_id(self, agent_state):
        ''' Predict the action for evaluation purpose.

        Args:
            agent_state (numpy.array): current state

        Returns:
            q_values_by_action_id (dict): A dictionary containing q_values by action_id
        '''
        q_values = self.predict(agent_state)
        q_values_by_action_id = {agent_state['raw_agent_actions'][i]: float(q_values[list(agent_state['agent_actions'].keys())[i]]) for i in range(len(agent_state['agent_actions']))}
        return q_values_by_action_id

    def feed(self, transition): # 230506
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            transition (list): a list of 6 elements that represent the transition
            The state and the next_state are the observations for this agent.
        '''
        (state, action, reward, next_state, done, next_legal_actions) = tuple(transition)
        self.feed_memory(state, action, reward, next_state, next_legal_actions, done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately,
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
        '''
        torch.save(self.checkpoint_attributes(), path + '/' + filename)

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''

        return {
            'agent_type': 'DQNAgent_230510',
            'q_estimator': self.q_estimator.checkpoint_attributes(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'epsilon_start': self.epsilons.min(),
            'epsilon_end': self.epsilons.max(),
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'discount_factor': self.discount_factor,
            'update_target_estimator_every': self.update_target_estimator_every,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device,
            # FIXED: 230510
            'use_raw': self.use_raw,
            'replay_memory_init_size': self.replay_memory_init_size,
            'model_name': self.model_name,
            'save_path': self.save_path,
            'save_every': self.save_every
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint

        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''

        print("\nINFO - Restoring model from checkpoint...")
        config = DQNAgentConfig()

        config.replay_memory_size = checkpoint['memory']['memory_size']
        config.replay_memory_init_size = checkpoint['replay_memory_init_size']
        config.update_target_estimator_every = checkpoint['update_target_estimator_every']
        config.discount_factor = checkpoint['discount_factor']
        config.epsilon_start = checkpoint['epsilon_start']
        config.epsilon_end = checkpoint['epsilon_end']
        config.epsilon_decay_steps = checkpoint['epsilon_decay_steps']
        config.batch_size = checkpoint['batch_size']
        config.train_every = checkpoint['train_every']
        config.save_every = checkpoint['save_every']
        config.learning_rate = checkpoint['q_estimator']['learning_rate']
        config.num_actions = checkpoint['num_actions']
        config.state_shape = checkpoint['q_estimator']['state_shape']
        config.mlp_layers = checkpoint['q_estimator']['mlp_layers']
        config.model_name = checkpoint['model_name']
        config.save_path = checkpoint['save_path']
        config.use_raw = checkpoint['use_raw']
        config.device = checkpoint['device']

        agent_instance = cls(config=config)

        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']

        agent_instance.q_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
        agent_instance.target_estimator = deepcopy(agent_instance.q_estimator)
        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

        return agent_instance
