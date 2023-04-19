import panel as pn
from collections import OrderedDict

from rlcard.games.gin_rummy.utils.action_event import ActionEvent

from World import World
from util import sortByRankBySuit

class ReviewPlayWindow(pn.Column):

    def __init__(self, world: World):
        super().__init__()
        self.world = world

        self.agents = [self.world.agent] + self.world.opponents

        self.review_play_pane = ReviewPlayPane(world=world)
        self.review_play_pane.margin = 5
        self.game_pane = self.review_play_pane.game_pane
        self.info_table_pane = self.review_play_pane.info_table_pane
        self.play_match_button = self.review_play_pane.play_match_control.play_match_button
        self.max_review_episodes_input = self.review_play_pane.play_match_control.max_review_episodes_input
        self.row_slider = self.review_play_pane.row_slider
        self.row_discrete_player = self.review_play_pane.row_discrete_player

        self.append(self.review_play_pane)

        self.background = 'green'

        ### hook up row_slider
        self.row_slider.param.watch(self.update_info_table_pane_by_event, 'value')

        ### hook up row_discrete_player
        self.row_discrete_player.param.watch(self.update_row_discrete_player_by_event, 'value')

        ### hook up play_review_match_button
        self.play_match_button.on_click(self.update_window)
    
    def update_info_table_pane_by_event(self, event):
        row_id = event.new
        self.row_discrete_player.value = row_id
        self.update_game_pane_by_row_id(row_id=row_id)
        self.update_info_table_pane_by_row_id(row_id=row_id)

    def update_row_discrete_player_by_event(self, event):
        row_id = event.new
        self.row_slider.value = row_id
        self.update_game_pane_by_row_id(row_id=row_id)
        self.update_info_table_pane_by_row_id(row_id=row_id)

    def update_info_table_pane_by_row_id(self, row_id: int):
        self.info_table_pane.object = self.info_table_pane.make_info_table(dataframe=self.dataframe, row_id=row_id, agents=self.agents)
    
    def update_game_pane_by_row_id(self, row_id: int):
        self.game_pane.object = self.game_pane.make_game_text(dataframe=self.dataframe, row_id=row_id, agents=self.agents)

    def update_window(self, event): #### dataframe will change
        self.review_play_pane.play_match_control.play_match_status.object = f'Running {event.new}'

		### play review match; dataframe will change
        self.world.play_review_match(max_review_episodes=self.max_review_episodes_input.value)

        max_rows = len(self.dataframe)

		### update row_slider
        self.row_slider.start = 0
        self.row_slider.value = 0
        self.row_slider.end = max_rows - 1
        
        ### update row_discrete_player
        self.row_discrete_player.value = 0
        self.row_discrete_player.options = list(range(max_rows))
        
        ### update game_pane
        self.update_game_pane_by_row_id(row_id=0)
        
        ### update info_table_pane
        self.update_info_table_pane_by_row_id(row_id=0)

        self.review_play_pane.play_match_control.play_match_status.object = f'Finished {event.new}'

    @property
    def dataframe(self):
        return self.world.gameReviewer.game_match.dataframe

class ReviewPlayPane(pn.Column):

    def __init__(self, world: World):
        super().__init__()
        agents = [world.agent] + world.opponents
        dataframe = world.gameReviewer.game_match.dataframe
        max_rows = len(dataframe)
        title_pane = pn.pane.Markdown(f"### Review Play Games")
        self.play_match_control = PlayMatchControl(max_review_episodes=30)
        self.select_using_filter_by_groups = SelectByGroups()
        self.row_slider = pn.widgets.IntSlider(name='row', start=0, end=max_rows - 1)
        self.row_discrete_player = pn.widgets.DiscretePlayer(name='Row', options=list(range(max_rows)), value=0, loop_policy='once', interval=1000 * 2)
        self.info_table_pane = InfoTablePane(dataframe=dataframe, row_id=2, agents=agents) 
        self.game_pane = GamePane(dataframe=dataframe, row_id=0, agents=agents)
        self.append(title_pane)
        self.append(self.play_match_control)
        self.append(self.select_using_filter_by_groups)
        self.append(self.row_slider)
        self.append(self.row_discrete_player)
        self.append(self.game_pane)
        self.append(self.info_table_pane)
        self.row_slider.max_width = 600
        self.row_discrete_player.max_width = self.row_slider.max_width

class PlayMatchControl(pn.Row):

    def __init__(self, max_review_episodes:int):
        super().__init__()
        self.play_match_status = pn.pane.Markdown()
        self.max_review_episodes_input = pn.widgets.IntInput(name='max episodes', value=max_review_episodes, start=0, end=1000)
        self.play_match_button = pn.widgets.Button(name='Play match')
        self.play_match_button.margin = [20, 0, 0, 0]
        self.max_review_episodes_input.margin = [0, 0, 0, 10]
        self.play_match_button.width = 140
        self.max_review_episodes_input.width = 100
        self.append(self.play_match_button)
        self.append(self.max_review_episodes_input)
        self.append(self.play_match_status)

class SelectByGroups(pn.widgets.Select):

    def __init__(self):
        super().__init__()
        filter_by_groups = {
            'Default': [
                "None"
            ],
            'Draw': [
                "Draw from discard pile does not make or extend a meld",
                "Draw from stock pile when draw from discard pile would make or extend a meld",
            ],
            'Discard': [
                "Discard card from a meld",
                "Discard card from a good pair",
                "Discard card from a poor pair",
            ],
            'Going out': [
                "Fail to knock when can",
                "Fail to gin when can",
            ]
        }
        self.name = "Filter by groups"
        self.groups = filter_by_groups
        self.width = 550

class GamePane(pn.pane.Markdown):

    def __init__(self, dataframe, row_id, agents):
        super().__init__()
        self.object = self.make_game_text(dataframe, row_id, agents)

    def make_game_text(self, dataframe, row_id, agents) -> str:
            max_rows = len(dataframe)
            if max_rows == 0:
                return "### Gin Rummy (no rows)"
            row = dataframe.iloc[row_id]
            current_row = row_id + 1
            player_id = row['player_id']
            top_card = row['top_card']
            top_card_name = top_card[0] if top_card else '*'
            legal_action_ids = row['legal_action_ids']
            action_id = row['action_id']

            held_cards = sorted(row['held_cards'], key=sortByRankBySuit, reverse=True)
            held_card_names = " ".join(map(str, held_cards))
            actionEvent = ActionEvent.decode_action(action_id)

            opponent_known_cards = sorted(row['opponent_known_cards'], key=sortByRankBySuit, reverse=True)
            opponent_known_cards_names = " ".join(map(str, opponent_known_cards))

            dead_cards = sorted(row['dead_cards'], key=sortByRankBySuit, reverse=True)
            dead_cards_names = " ".join(map(str, dead_cards))

            unknown_cards = sorted(row['unknown_cards'], key=sortByRankBySuit, reverse=True)
            unknown_cards_names = " ".join(map(str, unknown_cards))

            env_state = row['env_state']
            agent = agents[player_id]
            best_action, info = agent.eval_step(state=env_state)

            deck_count = len(held_cards) + len(dead_cards) + len(unknown_cards) + len(opponent_known_cards) + (1 if top_card else 0)
            if action_id == 0 or action_id == 1:
                deck_count += 1 # kludge

            return f"""
            ### Gin Rummy ({current_row} of {max_rows})

            player_id: {player_id}

            opponent_known_cards: [{opponent_known_cards_names}]

            top_card: {top_card_name} dead_cards: [{dead_cards_names}]

            held_cards: [{held_card_names}]

            legal_actions: {legal_action_ids}

            action: {actionEvent} <{action_id}>

            unknown_cards: [{unknown_cards_names}]

            deck_count: {deck_count} Note: should be 52.

            best_action: {best_action} {ActionEvent.decode_action(best_action)}
        """

class InfoTablePane(pn.pane.Markdown):

    def __init__(self, dataframe, row_id, agents):
        super().__init__()
        self.object = self.make_info_table(dataframe, row_id, agents)

    def make_info_table(self, dataframe, row_id:int, agents) -> str:
        max_rows = len(dataframe)
        if max_rows == 0:
            return ""
        row = dataframe.iloc[row_id]
        player_id = row['player_id']
        env_state = row['env_state']
        agent = agents[player_id]
        best_action, info = agent.eval_step(state=env_state)
        if not info:
            return ""
        info_sorted_by_value = OrderedDict(sorted(info['values'].items(), key=lambda x: x[1], reverse=True))
        info_table_lines = []
        info_table_lines.append("### DQNAgent\n")
        info_table_lines.append("<div class='orange_border_table'></div>")
        info_table_lines.append("| Action | Description | Q_value |")
        info_table_lines.append("\n")
        info_table_lines.append("| :--- | :--- | :---- |")
        info_table_lines.append("\n")
        for info_action, q_value in info_sorted_by_value.items():
            info_table_lines.append(f'|{info_action} | {ActionEvent.decode_action(info_action)} | {q_value} |')
            info_table_lines.append("\n")
        info_table = " ".join(info_table_lines)
        return info_table