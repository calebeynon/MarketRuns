from otree.api import *
import random

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
        
        # If no payoffs recorded, use a default
        if not pay_list:
            selected_payoff = 0
            selected_app = 0
        else:
            # Randomly select an index
            selected_index = random.randint(0, len(pay_list) - 1)
            selected_payoff = pay_list[selected_index]
            # App number is index + 1 (since apps are numbered 1-4)
            selected_app = selected_index + 1
        
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
        
        return {
            'selected_payoff': selected_payoff,
            'selected_app': selected_app,
            'participation_bonus': participation_bonus,
            'total_payment': total_payment
        }

page_sequence = [Final]