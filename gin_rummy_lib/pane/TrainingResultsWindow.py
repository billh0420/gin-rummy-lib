import os
import panel as pn
import pandas as pd
import csv

import matplotlib.pyplot as plt

from World import World

class TrainingResultsWindow(pn.Column):

    def __init__(self, world: World):
        super().__init__()
        self.agent_dir = world.agent_dir

        # Log
        self.log_title = pn.pane.Markdown("### Log")
        self.log_view = LogView(agent_dir=world.agent_dir)
        log_cell = pn.Column(self.log_title, self.log_view)

        # Reward by Episode
        self.fig_title = pn.pane.Markdown("### Reward by Episode")
        self.fig_view = self.get_fig_view()

        # Cumulated Reward by Episode
        cumulated_reward_fig_title = pn.pane.Markdown('### Cumulated Reward by Episode')
        cumulated_performance_csv_path = f'{world.agent_dir}/cumulated_performance.csv'
        cumulated_reward_fig_pane = TrainingResultsWindow.make_fig_pane(cumulated_performance_csv_path, algorithm='dqn', scale_factor=1)
        cumulated_reward_fig_view = pn.Column(cumulated_reward_fig_title, cumulated_reward_fig_pane)

        # Performance view
        self.performance_view = self.get_performance_view()
        self.performance_title = pn.pane.Markdown("### Performance")
        performance_cell = pn.Column(self.performance_title, self.performance_view)
        fig_cell = pn.Column(self.fig_title, self.fig_view)

        # Window
        content = pn.Row(pn.Column(log_cell, performance_cell), pn.Column(fig_cell, cumulated_reward_fig_view))
        self.append(content)
        self.background = "green"
        self.width_policy = 'max'

        # margins
        log_cell.margin = [0, 10, 10, 10]
        performance_cell.margin = [0, 10, 10, 10]
        fig_cell.margin=[0, 10, 10, 10]
        content.margin = [0, 0, 0, 0]

        # layout
        log_cell.width_policy = 'max'
        content.width_policy = 'max'

    def get_fig_view(self):
        file_path = f'{self.agent_dir}/fig.png'
        if not os.path.exists(file_path):
            fig_view = pn.pane.Str("No figure png")
            return fig_view
        fig_view = pn.pane.PNG(file_path)
        fig_view.width = 400
        return fig_view

    def get_performance_view(self):
        file_path = f'{self.agent_dir}/performance.csv'
        if not os.path.exists(file_path):
            performance_view = pn.pane.Str("No performance csv")
            return performance_view
        performance_df = pd.read_csv(file_path)
        performance_view = pn.widgets.Tabulator(performance_df, layout='fit_data_stretch')
        performance_view.height = 400
        return performance_view

    @staticmethod
    def make_fig(csv_path, algorithm):
        ''' Read data from csv file and return its plot
        '''
        max_episodes = 60 # plot only last number of episodes
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            xs = []
            ys = []
            for row in reader:
                xs.append(int(row['episode']))
                ys.append(float(row['reward']))
            fig, ax = plt.subplots()
            fig.set_figwidth(3 * 4.8)
            ax.plot(xs[-max_episodes:], ys[-max_episodes:], label=algorithm)
            ax.set(xlabel='episode', ylabel='reward')
            ax.legend()
            ax.grid()
        plt.close(fig)
        return fig

    @staticmethod
    def make_fig_pane(csv_path, algorithm, scale_factor:float = 1):
        if not os.path.exists(csv_path):
            fig_view = pn.pane.Str("No figure")
            return fig_view
        fig = TrainingResultsWindow.make_fig(csv_path=csv_path, algorithm=algorithm)
        fig_view = pn.pane.Matplotlib(fig, dpi=144 * scale_factor)
        return fig_view

class LogView(pn.Column):

    def __init__(self, agent_dir):
        super().__init__()
        file_path = f'{agent_dir}/log.txt'
        log_text = f"No log.{'':20}"
        if not os.path.exists(file_path):
            log_text = f"No log.{'':20}"
        else:
            with open(file_path, 'r') as file:
                log_text = file.read()
                if len(log_text) == 0:
                    log_text = f"No log.{'':20}"
        log_text_pane = pn.pane.Str(log_text)
        self.append(log_text_pane)
        self.scroll = True
        self.height = 400
        self.css_classes = ['log-widget-box']
        self.width_policy = 'max'