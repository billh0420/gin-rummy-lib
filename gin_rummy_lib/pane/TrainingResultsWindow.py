import os
import panel as pn
import pandas as pd

from World import World

class TrainingResultsWindow(pn.Column):

    def __init__(self, world: World):
        super().__init__()
        self.agent_dir = world.agent_dir
        self.log_view = self.get_log_view()
        self.fig_view = self.get_fig_view()
        self.performance_view = self.get_performance_view()

        self.log_title = pn.pane.Markdown("### Log")
        self.fig_title = pn.pane.Markdown("### Reward by episode")
        self.performance_title = pn.pane.Markdown("### Performance")

        log_cell = pn.Column(self.log_title, self.log_view, margin=[0, 10, 10, 10], width_policy='max')
        performance_cell = pn.Column(self.performance_title, self.performance_view, margin=[0, 10, 10, 10])
        fig_cell = pn.Column(self.fig_title, self.fig_view, margin=[0, 10, 10, 10])

        self.append(pn.Row(log_cell, fig_cell))
        self.append(performance_cell)
        self.background = "green"
        self.width_policy = 'max'
    
    def get_log_view(self):
        file_path = f'{self.agent_dir}/log.txt'
        if not os.path.exists(file_path):
            log_view = pn.pane.Str("No log text")
            return log_view
        with open(file_path, 'r') as file:
            log_txt = file.read()
        log_txt_pane = pn.pane.Str(log_txt)
        log_view = pn.Column(log_txt_pane, scroll=True, css_classes=['log-widget-box'])
        log_view.height = 400
        log_view.width_policy = 'max'
        return log_view
    
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