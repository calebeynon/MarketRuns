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

class Quiz(Page):
    pass


page_sequence = [ Quiz]