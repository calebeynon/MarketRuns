from otree.api import *
import numpy as np
import random

class C(BaseConstants):
    NAME_IN_URL = 'game'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = min(np.random.geometric(p=0.125), 14)  # Geometric distribution with mean 8
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

#FUNCTIONS
def set_all_false(group: Group):
    players = group.get_players()
    for p in players:
        p.participant.vars['sold'] = False
        p.participant.vars['payoff'] = 0
        p.participant.vars['price'] = C.INITIAL_PRICE
        p.participant.vars['signal'] = C.INITIAL_SIGNAL

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
            p.participant.vars['payoff'] = 2*a_price

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


# PAGES
class ChatWait(WaitPage):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    after_all_players_arrive = set_all_false


class Chat(Page):
    def is_displayed(player):
        return player.round_number == 1

    def get_timeout_seconds(player):
        return 45

class MarketPeriodWait(WaitPage):
    pass

class MarketPeriod(Page):
    def get_timeout_seconds(player):
        return 4
    form_model = 'player'

    @staticmethod
    def get_form_fields(player: Player):
        return ['sold']

    def vars_for_template(player):
        return {
            'sold_status': player.participant.vars['sold'],
            'price': player.participant.vars['price'],
            'payoff': player.participant.vars['payoff'],    
            'signal': player.participant.vars['signal']
        }
    

class PeriodResultsWait(WaitPage):
    after_all_players_arrive = set_payoffs

class PeriodResults(Page):
    def get_timeout_seconds(player):
        return 5
    def vars_for_template(player):
        sellers = [p.id_in_group for p in player.group.get_players() if p.participant.vars['sold']]
        last_round_sellers = [p.id_in_group for p in player.group.get_players() if p.sold]
        return {
            'sold_status': player.participant.vars['sold'],
            'payoff': player.participant.vars['payoff'],
            'sellers': sellers,
            'last_round_sellers': last_round_sellers,
            'signal': player.participant.vars['signal']
        }
    
    @staticmethod
    def before_next_page(player, timeout_happened):
        set_signal(player.group)
    
class ResultsWait(WaitPage):
    after_all_players_arrive = final_sale

class Results(Page):
    def get_timeout_seconds(player):
        return 30
    
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS
    
    def vars_for_template(player):
        return {
            'sold_status': player.participant.vars['sold'],
            'payoff': player.participant.vars['payoff']
        }

page_sequence = [ChatWait, Chat, MarketPeriodWait, MarketPeriod, PeriodResultsWait, PeriodResults, ResultsWait, Results]






