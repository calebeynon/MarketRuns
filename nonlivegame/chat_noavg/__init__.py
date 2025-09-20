from otree.api import *
import random

class C(BaseConstants):
    NAME_IN_URL = 'chat_noavg'
    PLAYERS_PER_GROUP = 4
    SEGMENT_NUMBER = 1  # This is the first segment
    
    # Generate random number of rounds for this segment
    NUM_ROUNDS_IN_SEGMENT = 2 #min(np.random.geometric(p=0.125), 14)  # 1-14 rounds
    
    # For each round, generate random number of periods
    PERIODS_PER_ROUND = []
    for round_num in range(NUM_ROUNDS_IN_SEGMENT):
        #periods = min(np.random.geometric(p=0.125), 14)  # 1-14 periods per round
        periods = 2
        PERIODS_PER_ROUND.append(periods)
    
    # Total number of oTree rounds (sum of all periods across all rounds)
    NUM_ROUNDS = sum(PERIODS_PER_ROUND)
    
    INITIAL_PRICE = 8
    STATE = [random.randint(0, 1) for _ in range(NUM_ROUNDS_IN_SEGMENT)]
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
    state = models.BooleanField(initial=C.STATE[0])
    round_number_in_segment = models.IntegerField()
    period_in_round = models.IntegerField()

    # Store payoffs for each custom round (up to 14 rounds per segment)
    round_1_payoff = models.FloatField(initial=0)
    round_2_payoff = models.FloatField(initial=0)
    round_3_payoff = models.FloatField(initial=0)
    round_4_payoff = models.FloatField(initial=0)
    round_5_payoff = models.FloatField(initial=0)
    round_6_payoff = models.FloatField(initial=0)
    round_7_payoff = models.FloatField(initial=0)
    round_8_payoff = models.FloatField(initial=0)
    round_9_payoff = models.FloatField(initial=0)
    round_10_payoff = models.FloatField(initial=0)
    round_11_payoff = models.FloatField(initial=0)
    round_12_payoff = models.FloatField(initial=0)
    round_13_payoff = models.FloatField(initial=0)
    round_14_payoff = models.FloatField(initial=0)

    def set_round_payoff(self, round_number, payoff):
        """Helper method to set payoff for a specific custom round"""
        field_name = f'round_{round_number}_payoff'
        setattr(self, field_name, payoff)

    def get_round_payoff(self, round_number):
        """Helper method to get payoff for a specific custom round"""
        field_name = f'round_{round_number}_payoff'
        return getattr(self, field_name, 0)

#FUNCTIONS
def creating_session(subsession: Subsession):
    grouping = [[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]
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
        
        # Initialize pay_list only on the very first oTree round
        if subsession.round_number == 1:
            p.participant.vars['pay_list'] = []
            p.participant.vars['pay_list_random'] = []

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
        random.shuffle(sellers)        
        pays = [a_price - 2*i for i in range(len(sellers))] if len(sellers) > 1 else [a_price] if len(sellers) == 1 else []
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
    if C.STATE[players[0].round_number_in_segment - 1] == 0:
        random.shuffle(non_sellers)
        pays = [a_price - 2*i for i in range(len(non_sellers))] if len(non_sellers) > 1 else [a_price] if len(non_sellers) == 1 else []
        for i, p in enumerate(non_sellers):
            p.participant.vars['payoff'] = pays[i]
            p.payoff = pays[i]
    else:
        for p in non_sellers:
            p.participant.vars['payoff'] = 20
            p.payoff = 20

def set_signal(group: Group):
    players = group.get_players()
    a_signal = players[0].participant.vars['signal']
    if C.STATE[players[0].round_number_in_segment - 1]:
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

class SegmentIntroWait1(WaitPage):
    wait_for_all_groups = True
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1 and player.period_in_round == 1

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
        return 8
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
        # Get all unique sellers from current period and previous custom rounds
        all_sellers = set()

        # Get sellers from current period
        for p in player.group.get_players():
            if p.participant.vars['sold']:
                all_sellers.add(p.participant.label)

        # Get sellers from all previous custom rounds in this segment
        for prev_round_num in range(1, player.round_number_in_segment):
            for p in player.group.get_players():
                # Check if this player sold in that specific custom round
                if p.get_round_payoff(prev_round_num) > 0:
                    all_sellers.add(p.participant.label)
        
        return {
            'sold_status': player.participant.vars['sold'],
            'price': player.participant.vars['price'],
            'payoff': player.participant.vars['payoff'],    
            'signal': int(round(player.participant.vars['signal'] * 100)),
            'signal_history': list(player.participant.vars['signal_history']),
            'price_history': list(player.participant.vars['price_history']),
            'all_sellers': sorted(list(all_sellers)),
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round
        }
    
class MarketPeriodPayoffWait(WaitPage):
    @staticmethod
    def after_all_players_arrive(group):
        # Calculate payoffs based on who sold
        set_payoffs(group)

        # Update player model fields for all players
        for player in group.get_players():
            # Store round-specific payoff
            if player.participant.vars['sold']:
                round_payoff = player.participant.vars['payoff']
                player.set_round_payoff(player.round_number_in_segment, round_payoff)
                player.payoff = round_payoff  # Still set current payoff for oTree
            else:
                player.set_round_payoff(player.round_number_in_segment, 0)
                player.payoff = 0

            # Update other fields
            player.signal = player.participant.vars['signal']
            player.price = player.participant.vars['price']
            player.state = C.STATE[player.round_number_in_segment - 1]
            player.sold = player.participant.vars['sold']


class ResultsWait(WaitPage):
    wait_for_all_groups = True
    @staticmethod
    def after_all_players_arrive(subsession):
        # Apply final sale logic to all groups
        for group in subsession.get_groups():
            final_sale(group)

            # Update round payoffs after final sale
            for player in group.get_players():
                final_payoff = player.participant.vars['payoff']
                player.set_round_payoff(player.round_number_in_segment, final_payoff)
                player.payoff = final_payoff

class Results(Page):
    def get_timeout_seconds(player):
        return 30
    
    def is_displayed(player):
        # Display only at the end of each round (when period equals total periods in round)
        periods_in_current_round = C.PERIODS_PER_ROUND[player.round_number_in_segment - 1]
        return player.period_in_round == periods_in_current_round
    
    def before_next_page(player, timeout_happened):
        # Use the stored round payoff instead of participant.vars
        round_payoff = player.get_round_payoff(player.round_number_in_segment)
        player.participant.vars['pay_list'].append(round_payoff)

        # Only add a random payoff at the end of each segment
        if player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT:
            # Get all payoffs from this segment and randomly select one
            segment_payoffs = player.participant.vars['pay_list'][-C.NUM_ROUNDS_IN_SEGMENT:]  # Last N payoffs
            player.participant.vars['pay_list_random'].append(random.choice(segment_payoffs))

    def vars_for_template(player):
        # Show 8 if it's the final round, otherwise random 1-7
        if player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT:
            display_number = 8
        else:
            display_number = random.randint(1, 7)

        # Use the stored round payoff
        round_payoff = player.get_round_payoff(player.round_number_in_segment)

        return {
            'sold_status': player.participant.vars['sold'],
            'payoff': round_payoff,
            'segment_number': C.SEGMENT_NUMBER,
            'round_number': player.round_number_in_segment,
            'period_number': player.period_in_round,
            'display_number': display_number,
            'is_final_round': player.round_number_in_segment == C.NUM_ROUNDS_IN_SEGMENT
        }

page_sequence = [SegmentIntroWait1, SegmentIntro, SegmentIntroWait, MarketPeriodWait, MarketPeriod, MarketPeriodPayoffWait, ResultsWait, Results]






