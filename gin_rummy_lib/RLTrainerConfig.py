class RLTrainerConfig:

    def __init__(self):
        self.algorithm = 'dqn'
        self.num_episodes = 20000
        self.num_eval_games = 100
        self.evaluate_every = max(1, min(self.num_episodes // 20, 10000))

    def __str__(self):
        lines = []
        lines.append(f'----- RLTrainerConfig -----')
        lines.append(f'algorithm={self.algorithm}')
        lines.append(f'num_episodes={self.num_episodes}')
        lines.append(f'num_eval_games={self.num_eval_games}')
        lines.append(f'evaluate_every={self.evaluate_every}')
        return "\n".join(lines)