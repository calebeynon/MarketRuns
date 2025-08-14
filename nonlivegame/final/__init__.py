from otree.api import *
import random
import numpy as np

class C(BaseConstants):
    NAME_IN_URL = 'final'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    selected_payoff = models.FloatField()
    selected_app = models.IntegerField()
    participation_bonus = models.FloatField(initial=7.50)
    total_payment = models.FloatField()

class Final(Page):
    def vars_for_template(player):
        # Get payoff list from participant vars
        pay_list = player.participant.vars.get('pay_list', [])
        pay_list_random = player.participant.vars.get('pay_list_random', [])
        
        # If no payoffs recorded, use a default
        if not pay_list:
            selected_payoff = 0
            selected_app = 0
        else:
            # Sum all payoffs from pay_list
            selected_payoff = sum(pay_list_random)
            # App number is set to 0 since we're summing all apps
            selected_app = 0
        
        # Calculate total payment
        participation_bonus = 7.50
        total_payment = selected_payoff + participation_bonus
        
        # Store in player fields
        player.selected_payoff = selected_payoff
        player.selected_app = selected_app
        player.participation_bonus = participation_bonus
        player.total_payment = total_payment
        
        # Set as final payoff
        player.payoff = total_payment
        
        # Calculate real-world currency values
        # Conversion rate is 0.025 USD per point
        selected_payoff_usd = selected_payoff * 0.25
        participation_bonus_usd = participation_bonus
        total_payment_usd = selected_payoff_usd + participation_bonus_usd
        
        return {
            'selected_payoff': selected_payoff,
            'selected_app': selected_app,
            'participation_bonus': participation_bonus,
            'total_payment': total_payment,
            'selected_payoff_usd': np.round(selected_payoff_usd, 2),
            'participation_bonus_usd': np.round(participation_bonus_usd, 2),
            'total_payment_usd': np.round(total_payment_usd, 2)
        }

page_sequence = [Final]