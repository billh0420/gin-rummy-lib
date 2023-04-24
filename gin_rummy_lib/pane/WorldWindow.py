import panel as pn
import torch
import pathlib
import glob

from World import World
from pane.DQNAgentPane import DQNAgentPane

class WorldWindow(pn.Column):

    def __init__(self, world: World):
        super().__init__()
        self.world = world

        self.training_agent_paths = glob.glob(f'{self.world.world_dir}/agents/*/*.pth', recursive=True)
        self.training_agent_names = [pathlib.Path(x).resolve().stem for x in self.training_agent_paths]
        self.training_agents = [torch.load(agent_path) for agent_path in self.training_agent_paths]
        self.traing_agents_by_name = dict(zip(self.training_agent_names, self.training_agents))

        # selection view
        self.training_agents_list = pn.widgets.Select(name='Training agent', options=sorted(self.training_agent_names))
        self.opponents_list = pn.widgets.Select(name='Opponent', options=self.world.opponent_names, value=self.world.opponent_name)
        player_selection_view = pn.Column(self.training_agents_list, self.opponents_list)

        # agent settings
        agent_settings_title = pn.pane.Markdown(f'## Agent Settings')
        selected_agent_name = self.training_agents_list.value
        selected_agent = self.traing_agents_by_name[selected_agent_name]
        self.agent_pane = DQNAgentPane(dqn_agent=selected_agent)
        agent_settings_view = pn.Column(agent_settings_title, self.agent_pane)

        # hook up
        self.training_agents_list.param.watch(self.on_select_training_agent, 'value')
        self.opponents_list.param.watch(self.on_select_opponent, 'value')

        # window
        window_title = pn.pane.Markdown('# World Window')
        body = pn.Row(player_selection_view, agent_settings_view)
        window_content = pn.Column(window_title, body)
        self.append(window_content)

        # margins
        player_selection_view.margin = [25, 0, 0, 0]
        self.training_agents_list.margin = [0, 0, 0, 0]
        self.opponents_list.margin = [10, 0, 0, 0]

        agent_settings_title.margin = 0
        self.agent_pane.margin = [0, 0, 0, 0]
        agent_settings_view.margin = [0, 0, 0, 20]

        window_title.margin = 0
        body.margin = 0
        window_content.margin = [0, 10, 10, 10]

        # layout
        agent_settings_view.width_policy = 'max'
        window_content.width_policy = 'max'
        self.width_policy = 'max'

        # background
        self.background = 'green'

    def on_select_training_agent(self, event):
        old_selected_agent_name = event.old
        selected_agent_name = event.new
        if selected_agent_name != old_selected_agent_name:
            self.world.model_name = selected_agent_name
            selected_agent = self.traing_agents_by_name[selected_agent_name]
            self.agent_pane.object = DQNAgentPane(dqn_agent=selected_agent).object

    def on_select_opponent(self, event):
        old_selected_opponent_name = event.old
        selected_opponent_name = event.new
        if selected_opponent_name != old_selected_opponent_name:
            self.world.opponent_name = selected_opponent_name