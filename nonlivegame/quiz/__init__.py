from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'quiz'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Player(BasePlayer):
    pass

class Group(BaseGroup):
    pass

class Instructions(Page):
    def vars_for_template(player):
        return {
            'id': player.participant.label
        }

class Quiz(Page):
    pass


page_sequence = [Instructions, Quiz]