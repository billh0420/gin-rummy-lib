from typing import List

import more_itertools as mit
from traitlets.traitlets import Int

from datetime import datetime
from pytz import timezone

import re # regular expression

import rlcard.games.gin_rummy.utils.utils as gin_rummy_utils

from rlcard.games.base import Card
from rlcard.games.gin_rummy.utils.utils import get_rank_id, get_suit_id

def get_current_time():
    return datetime.now(timezone('America/Chicago')).strftime('%I:%M:%S %p')

def to_int_list(input: str, default: List[int] or None = None) -> List[int]:
    result = []
    pattern = r"""(?x)          # ignore pattern whitespace
        ^                       # start of string
        \s*                     # whitespace
        \[                      # opening bracket
        \s*                     # whitespace
        (\d+(\s*,\s*\d+)*)*     # zero or more numbers seperated by commas
        \s*                     # whitespace
        \]                      # closing bracket
        \s*                     # whitespace
        $                       # end of string
    """
    if re.match(pattern, input):
        result = list(map(int, re.findall('[0-9]+', input)))
    if not result:
        if isinstance(default, list) and set(map(type, default)) == {int}:
            result = default
        else:
            raise ValueError(f'to_int_list: {input}')
    return result

def sortByRankBySuit(card: Card):
    # by rank (A 2 3 ... J Q K) by suit (C D H S)
    return 4 * get_rank_id(card) + 3 - get_suit_id(card)

def ob_to_num(ob):
    # ob is a numpy array
    sum:Int = 2 ** 52
    summand = 1
    for feature in ob:
        if feature == 1:
            sum += summand
        summand *= 2
    return sum

def num_to_ob(number:int):
    phi = reversed(list(f'{number:b}')[1:])
    card_ids = [int(x) for x in phi]
    return card_ids

def get_player_row(state, action:int):
    obs = state['obs']
    held_cards_num = ob_to_num(obs[0])
    top_card_num = ob_to_num(obs[1])
    dead_cards_num = ob_to_num(obs[2])
    opponent_known_cards_num = ob_to_num(obs[3])
    unknown_cards_num = ob_to_num(obs[4])
    encoded_held_pile = obs[0]
    encoded_top_card = obs[1]
    legal_actions = sorted(list(state['raw_legal_actions']))
    held_pile = gin_rummy_utils.decode_cards(encoded_held_pile)
    top_card = mit.first(gin_rummy_utils.decode_cards(encoded_top_card), default=None)
    row = {'held_cards_num': held_cards_num,
            'top_card_num': top_card_num,
            'dead_cards_num': dead_cards_num,
            'opponent_known_cards_num': opponent_known_cards_num,
            'unknown_cards_num': unknown_cards_num,
            'legal_actions': legal_actions,
            'action': action}
    return row

def get_player_rows(player_id:int, trajectories):
    result = []
    trajectory = trajectories[player_id]
    trajectory_count = len(trajectory) # should always be odd
    for index in range(trajectory_count // 2):
        state = trajectory[2 * index]
        action = trajectory[2 * index + 1]
        row = get_player_row(state=state, action=action)
        result.append(row)
    return result