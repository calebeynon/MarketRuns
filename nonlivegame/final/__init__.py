from otree.api import *
import random

class C(BaseConstants):
    NAME_IN_URL = 'final'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    ECUS_DOLLAR = 4

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    selected_payoff = models.FloatField()
    selected_app = models.IntegerField()
    participation_bonus = models.FloatField(initial=7.50)
    total_payment = models.FloatField()
    survey_bonus = models.FloatField()

class Final(Page):
    def vars_for_template(player):
        # Get payoff list from participant vars
        pay_list = player.participant.vars.get('pay_list', [])
        pay_list_random = player.participant.vars.get('pay_list_random', [])
        pay_list_random_index = player.participant.vars.get('pay_list_random_index', [])
        
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
        participation_bonus = 5.00
        survey_bonus = player.participant.vars['survey_payoff']
        total_payment = selected_payoff/C.ECUS_DOLLAR + participation_bonus + survey_bonus/C.ECUS_DOLLAR
        
        # Store in player fields
        player.selected_payoff = selected_payoff
        player.selected_app = selected_app
        player.participation_bonus = participation_bonus
        player.survey_bonus = survey_bonus
        player.total_payment = total_payment
        
        # Set as final payoff
        player.payoff = selected_payoff + survey_bonus #excluding participation bonus
        player.participant.payoff = selected_payoff + survey_bonus
        
        # Calculate real-world currency values
        # Conversion rate is 0.25 USD per point
        selected_payoff_usd = selected_payoff * 0.25
        participation_bonus_usd = participation_bonus
        survey_bonus_usd = survey_bonus * 0.25
        total_payment_usd = selected_payoff_usd + participation_bonus_usd + survey_bonus_usd
        
        return {
            'selected_payoff': selected_payoff,
            'selected_app': selected_app,
            'participation_bonus': participation_bonus,
            'total_payment': total_payment,
            'selected_payoff_usd': round(selected_payoff_usd, 2),
            'participation_bonus_usd': round(participation_bonus_usd, 2),
            'survey_bonus_usd': round(survey_bonus_usd, 2),
            'total_payment_usd': round(total_payment_usd, 2),
            'selected_one': pay_list_random_index[0] + 1,
            'selected_two': pay_list_random_index[1] + 1,
            'selected_three': pay_list_random_index[2] + 1,
            'selected_four': pay_list_random_index[3] + 1,
        }

page_sequence = [Final]