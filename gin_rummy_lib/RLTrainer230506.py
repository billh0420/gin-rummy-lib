from typing import List

import os
import torch

from rlcard.utils import Logger, plot_curve, get_device, set_seed

from rlcard.games.gin_rummy.game import GinRummyGame

from Contest import Contest, Contest_Delegate
from DQNAgent_230506 import Transition

class RLTrainer230506:

    # TODO:
    #   - where is device being used
    #   - is seed setting recognized by numpy, torch, random ???

    def __init__(self):
        self.training_agent_id = 0
        self.algorithm = 'dqn'

    def train(self, agents, game: GinRummyGame, num_episodes:int, num_eval_games: int = 100, seed = None):
        evaluate_every = max(1, min(num_episodes // 20, 10000))

        print(f'----- Overrides -----')
        print(f'actual num_episodes={num_episodes}')
        print(f'actual evaluate_every={evaluate_every}')
        print(f'actual num_eval_games={num_eval_games}')
        print(f'---------------------')

        # Check whether gpu is available
        device = get_device()
        # Seed numpy, torch, random
        set_seed(seed)
        agent = agents[self.training_agent_id]
        log_dir = self.get_log_dir(agent=agent)
        with Logger(log_dir=log_dir) as logger:
            for episode in range(num_episodes):
                # Feed transitions into agent memory, and train the agent.
                # We assume that DQNAgent always plays the first position.
                transitions = self.get_transitions(agents=agents, game=game)
                for transition in transitions:
                    feed_transition = list(transition)
                    agent.feed(feed_transition)
                # Evaluate the performance.
                if episode % evaluate_every == 0:
                    self.evaluate_performance(game=game, agents=agents, num_eval_games=num_eval_games, episode=episode, logger=logger)
        # Plot the learning curve
        self.plot_learning_curve(logger=logger)
        # Save model
        self.save_model(agent=agent)

    def get_model_name(self, agent):
        model_name = None
        save_path = agent.save_path
        if save_path is not None:
            model_name = os.path.basename(save_path)
        return model_name

    def get_log_dir(self, agent):
        return agent.save_path

    def evaluate_performance(self, game, agents, num_eval_games: int, episode: int, logger: Logger):
        agent = agents[self.training_agent_id]
        opponent = agents[1]
        contest_delegate = RLTrainerContestDelegate(episode=episode, num_eval_games=num_eval_games, training_agent_id=self.training_agent_id, logger=logger)
        contest = Contest(game=game, delegate=contest_delegate)
        contest.compete(num_rounds=num_eval_games, agent=agent, opponent=opponent)

    def plot_learning_curve(self, logger: Logger):
        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, self.algorithm)

    def save_model(self, agent):
        log_dir = self.get_log_dir(agent=agent)
        model_name = self.get_model_name(agent=agent)
        save_path = os.path.join(log_dir, f'{model_name}.pth')
        torch.save(agent, save_path)
        print('Model saved in', save_path)

    def get_transitions(self, agents, game: GinRummyGame) -> List[Transition]:
        # start a new game
        game.init_game()
        transitions:List[Transition] = []
        prior_agent_state = None
        prior_action_id:int = None
        tried_illegal_action = False
        while not game.round.is_over and not tried_illegal_action:
            # get current player_id
            player_id = game.round.current_player_id
            # get agent for current player_id
            agent = agents[player_id]
            # agent chooses action
            agent_state = agent.get_agent_state(player_id=player_id, game=game)
            action_id = agent.step(state=agent_state)
            game_action = game.decode_action(action_id=action_id)
            # update transitions
            if player_id == self.training_agent_id and prior_agent_state is not None:
                # legal_action_ids = [x.action_id for x in game.judge.get_legal_actions()]
                legal_actions = agent.get_legal_actions(player_id=player_id, game=game)
                transition = Transition(state=prior_agent_state['obs'], action=prior_action_id, reward=0, next_state=agent_state['obs'], done=False, legal_actions=legal_actions)
                transitions.append(transition)
            # execute action
            if action_id in [x.action_id for x in game.judge.get_legal_actions()]:
                game.step(action=game_action)
            else:
                tried_illegal_action = True
            # update transition
            if player_id == self.training_agent_id:
                prior_agent_state = agent_state
                prior_action_id = action_id
        payoffs = game.judge.scorer.get_payoffs(game=game)
        # update transitions for final transition
        if prior_agent_state is not None:
            reward = payoffs[self.training_agent_id]
            final_agent_state = agent.get_agent_state(player_id=self.training_agent_id, game=game)
            transition = Transition(state=prior_agent_state['obs'], action=prior_action_id, reward=reward, next_state=final_agent_state['obs'], done=True, legal_actions=[])
            transitions.append(transition)
        return transitions

class RLTrainerContestDelegate(Contest_Delegate):

    def __init__(self, episode, num_eval_games: int, training_agent_id: int, logger):
        self.num_eval_games = num_eval_games
        self.episode = episode
        self.training_agent_id = training_agent_id
        self.logger = logger
        self.contest_payoffs = [0, 0]

    def did_finish_game(self, game, agent, opponent):
        payoffs = game.judge.scorer.get_payoffs(game=game)
        self.contest_payoffs = list(map(sum, zip(self.contest_payoffs, payoffs)))

    def did_finish_competition(self, game, agent, opponent, num_rounds: int):
        agent_total_payoff = self.contest_payoffs[self.training_agent_id]
        agent_average_payoff = agent_total_payoff / self.num_eval_games
        self.logger.log_performance(self.episode, agent_average_payoff)
