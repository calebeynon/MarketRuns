from otree.api import *
import numpy as np
import random

class C(BaseConstants):
    NAME_IN_URL = 'chat_noavg2'
    PLAYERS_PER_GROUP = 4
    SEGMENT_NUMBER = 2  # This is the second segment
    
    # Generate random number of rounds for this segment
    NUM_ROUNDS_IN_SEGMENT = min(np.random.geometric(p=0.125), 14)  # 1-14 rounds
    
    # For each round, generate random number of periods
    PERIODS_PER_ROUND = []
    for round_num in range(NUM_ROUNDS_IN_SEGMENT):
        periods = min(np.random.geometric(p=0.125), 14)  # 1-14 periods per round
        PERIODS_PER_ROUND.append(periods)
    
    # Total number of oTree rounds (sum of all periods across all rounds)
    NUM_ROUNDS = sum(PERIODS_PER_ROUND)
    
    INITIAL_PRICE = 8
    STATE = np.random.randint(0, 2)
    PCORRECT = 0.675
    INITIAL_SIGNAL = 0.5

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    sold = models.BooleanField(initial=False)
    signal = models.FloatField(initial=C.INITIAL_SIGNAL)
    price = models.FloatField(initial=C.INITIAL_PRICE)
    state = models.BooleanField(initial=C.STATE)
    round_number_in_segment = models.IntegerField()
    period_in_round = models.IntegerField()

#FUNCTIONS
def creating_session(subsession: Subsession):
    grouping = [[1,6,11,16],[2,5,12,15],[3,8,9,14],[4,7,10,13]]
    subsession.set_group_matrix(grouping)
    
    # Calculate which round and period this oTree round belongs to
    current_otree_round = subsession.round_number
    cumulative_periods = 0
    round_num = 0
    period_in_round = 0
    
    # Find which round and period we're in
    for round_idx, periods_in_this_round in enumerate(C.PERIODS_PER_ROUND):
        if current_otree_round <= cumulative_periods + periods_in_this_round:
            round_num = round_idx + 1
            period_in_round = current_otree_round - cumulative_periods
            break
        cumulative_periods += periods_in_this_round
    
    # Set tracking for all players
    for p in subsession.get_players():
        p.round_number_in_segment = round_num
        p.period_in_round = period_in_round
        
        # Store the total periods in current round for display logic
        if 'periods_in_current_round' not in p.participant.vars:
            p.participant.vars['periods_in_current_round'] = {}
        p.participant.vars['periods_in_current_round'][round_num] = C.PERIODS_PER_ROUND[round_num - 1]
        
        # Initialize pay_list only on the very first round
        if subsession.round_number == 1:
            p.participant.vars['pay_list'] = []

def set_all_false(group: Group):
    players = group.get_players()
    for p in players:
        p.participant.vars['sold'] = False
        p.participant.vars['payoff'] = 0
        p.participant.vars['price'] = C.INITIAL_PRICE
        p.participant.vars['signal'] = C.INITIAL_SIGNAL
        p.participant.vars['signal_history'] = [C.INITIAL_SIGNAL]
        p.participant.vars['price_history'] = [C.INITIAL_PRICE]
        p.participant.vars['signal_history_length'] = 0
        p.participant.vars['price_history_length'] = 0

def set_payoffs(group: Group):
    players = group.get_players()
    sellers = [p for p in players if p.sold]
    for p in players:
        if p.sold:
            p.participant.vars['sold'] = True

    a_price = players[0].participant.vars['price']
    if len(sellers) > 0:
        np.random.shuffle(sellers)        
        pays = np.linspace(a_price, a_price - 2*len(sellers) + 2, len(sellers))
        for i,p in enumerate(players):
            p.participant.vars['price'] = a_price - 2*len(sellers)
        for i, p in enumerate(sellers):
            p.participant.vars['payoff'] = pays[i]
    
    # Update price history for all players regardless of whether there were sellers
    for p in players:
        p.participant.vars['price_history'].append(p.participant.vars['price'])
        p.participant.vars['price_history_length'] += 1

def final_sale(group: Group):
    players = group.get_players()
    a_price = players[0].participant.vars['price']
    non_sellers = [p for p in players if not p.participant.vars['sold']]
    if C.STATE == 0:
        np.random.shuffle(non_sellers)
        pays = np.linspace(a_price, a_price - 2*len(non_sellers) + 2, len(non_sellers))
        for i, p in enumerate(non_sellers):
            p.participant.vars['payoff'] = pays[i]
    else:
        for p in non_sellers:
            p.participant.vars['payoff'] = 20

def set_signal(group: Group):
    players = group.get_players()
    a_signal = players[0].participant.vars['signal']
    if C.STATE:
        signal = 1 if random.random() < C.PCORRECT else 0
        if signal:
            new_signal = (C.PCORRECT * a_signal) / ((C.PCORRECT * a_signal) + ((1 - C.PCORRECT) * (1 - a_signal)))
        else:
            new_signal = ((1 - C.PCORRECT) * a_signal) / (((1 - C.PCORRECT) * a_signal) + (C.PCORRECT * (1 - a_signal)))
    else:
        signal = 0 if random.random() < C.PCORRECT else 1
        if signal:
            new_signal = (C.PCORRECT * a_signal) / ((C.PCORRECT * a_signal) + ((1 - C.PCORRECT) * (1 - a_signal)))
        else:
            new_signal = ((1 - C.PCORRECT) * a_signal) / (((1 - C.PCORRECT) * a_signal) + (C.PCORRECT * (1 - a_signal)))
    for p in players:
        p.participant.vars['signal'] = new_signal
        p.participant.vars['signal_history'].append(new_signal)
        p.participant.vars['signal_history_length'] += 1

# PAGES
class SegmentIntro(Page):
    @staticmethod
    def is_displayed(player):
        # Only display at the very beginning of the segment
        return player.round_number == 1 and player.period_in_round == 1
    
    def get_timeout_seconds(player):
        return 15
    
    def vars_for_template(player):
        # Get group members
        group_members = [p.participant.label for p in player.group.get_players()]
        
        # Communication availability (segments 1 & 2 have no chat)
        has_communication = False
        
        return {
            'segment_number': C.SEGMENT_NUMBER,
            'total_segments': 4,
            'group_members': group_members,
            'has_communication': has_communication
        }

class SegmentIntroWait(WaitPage):
    @staticmethod
    def is_displayed(player):
        # Only display at the very beginning of the segment
        return player.round_number == 1 and player.period_in_round == 1

class ChatWait(WaitPage):
    @staticmethod
    def is_displayed(player):
        # Display at the start of each round
        return player.period_in_round == 1
    after_all_players_arrive = set_all_false


class Chat(Page):
    def is_displayed(player):
        # Display at the start of each round
        return player.period_in_round == 1

    def get_timeout_seconds(player):
        if player.round_number_in_segment == 1:
            return 90
        else:
            return 45
    
    def vars_for_template(player):
        return {
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round
        }

class MarketPeriodWait(WaitPage):
    @staticmethod
    def after_all_players_arrive(group):
        # Call set_all_false at the start of each round
        if group.get_players()[0].period_in_round == 1:
            set_all_false(group)
        
        # Reset sold status and payoff for all players at the beginning of each period
        for p in group.get_players():
            p.sold = False
            p.payoff = 0
        
        # Only update signal if not the first period of a round
        # Check using the first player's period_in_round
        if group.get_players()[0].period_in_round > 1:
            set_signal(group)

class MarketPeriod(Page):
    def get_timeout_seconds(player):
        return 4
    form_model = 'player'

    @staticmethod
    def get_form_fields(player: Player):
        return ['sold']

    @staticmethod
    def js_vars(player):
        return {
            'signal_history': list(player.participant.vars['signal_history']),
            'price_history': list(player.participant.vars['price_history']),
            'signal_history_length': player.participant.vars['signal_history_length'],
            'price_history_length': player.participant.vars['price_history_length']
        }

    def vars_for_template(player):
        return {
            'sold_status': player.participant.vars['sold'],
            'price': player.participant.vars['price'],
            'payoff': player.participant.vars['payoff'],    
            'signal': np.round(player.participant.vars['signal'], 2),
            'signal_history': list(player.participant.vars['signal_history']),
            'price_history': list(player.participant.vars['price_history']),
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round
        }
    

class PeriodResultsWait(WaitPage):
    after_all_players_arrive = set_payoffs

class PeriodResults(Page):
    def get_timeout_seconds(player):
        return 5
    def vars_for_template(player):
        sellers = [p.participant.label for p in player.group.get_players() if p.participant.vars['sold']]
        last_round_sellers = [p.participant.label for p in player.group.get_players() if p.sold]
        return {
            'sold_status': player.participant.vars['sold'],
            'payoff': player.participant.vars['payoff'],
            'sellers': sellers,
            'last_round_sellers': last_round_sellers,
            'signal': np.round(player.participant.vars['signal'], 2),
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round
        }
    def before_next_page(player, timeout_happened):
        # Only set player.payoff if the player has sold
        if player.participant.vars['sold']:
            player.payoff = player.participant.vars['payoff']
        else:
            player.payoff = 0
        player.signal = player.participant.vars['signal']
        player.price = player.participant.vars['price']
        player.sold = player.participant.vars['sold']
        player.state = C.STATE
    
class ResultsWait(WaitPage):
    after_all_players_arrive = final_sale

class Results(Page):
    def get_timeout_seconds(player):
        return 30
    
    def is_displayed(player):
        # Display only at the end of each round (when period equals total periods in round)
        periods_in_current_round = C.PERIODS_PER_ROUND[player.round_number_in_segment - 1]
        return player.period_in_round == periods_in_current_round
    
    def before_next_page(player, timeout_happened):
        player.participant.vars['pay_list'].append(player.participant.vars['payoff'])

    def vars_for_template(player):
        # Show 8 if it's the final round, otherwise random 1-7
        if player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT:
            display_number = 8
        else:
            display_number = random.randint(1, 7)
            
        return {
            'sold_status': player.participant.vars['sold'],
            'payoff': player.participant.vars['payoff'],
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round,
            'display_number': display_number,
            'is_final_round': player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT
        }

class RoundEnd(Page):
    def get_timeout_seconds(player):
        return 20
    
    def is_displayed(player):
        # Display only at the end of each round (when period equals total periods in round)
        periods_in_current_round = C.PERIODS_PER_ROUND[player.round_number_in_segment - 1]
        return player.period_in_round == periods_in_current_round
    
    def vars_for_template(player):
        # Show 8 if it's the final round, otherwise random 1-7
        if player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT:
            display_number = 8
        else:
            display_number = random.randint(1, 7)
        
        return {
            'display_number': display_number,
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round,
            'is_final_round': player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT
        }

page_sequence = [SegmentIntro, SegmentIntroWait, MarketPeriodWait, MarketPeriod, PeriodResultsWait, PeriodResults, ResultsWait, Results]






