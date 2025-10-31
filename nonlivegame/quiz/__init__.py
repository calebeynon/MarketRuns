from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'quiz'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Player(BasePlayer):
    # Store the participant's first selections for each quiz question
    first_q1 = models.StringField(blank=True, initial='')
    first_q2 = models.StringField(blank=True, initial='')
    first_q3 = models.StringField(blank=True, initial='')
    first_q4 = models.StringField(blank=True, initial='')
    first_q5 = models.StringField(blank=True, initial='')
    first_q6 = models.StringField(blank=True, initial='')
    first_q7 = models.StringField(blank=True, initial='')
    first_q8 = models.StringField(blank=True, initial='')
    first_q9 = models.StringField(blank=True, initial='')
    first_q10 = models.StringField(blank=True, initial='')
    first_q11 = models.StringField(blank=True, initial='')
    first_q12 = models.StringField(blank=True, initial='')
    first_q13 = models.StringField(blank=True, initial='')

class Group(BaseGroup):
    pass

class Instructions(Page):
    def vars_for_template(player):
        return {
            'id': player.participant.label
        }

class Label(Page):
    def vars_for_template(player):
        return {
            'id': player.participant.label,
            'first_answers': {
                'q1': player.field_maybe_none('first_q1') or '',
                'q2': player.field_maybe_none('first_q2') or '',
                'q3': player.field_maybe_none('first_q3') or '',
                'q4': player.field_maybe_none('first_q4') or '',
                'q5': player.field_maybe_none('first_q5') or '',
                'q6': player.field_maybe_none('first_q6') or '',
                'q7': player.field_maybe_none('first_q7') or '',
                'q8': player.field_maybe_none('first_q8') or '',
                'q9': player.field_maybe_none('first_q9') or '',
                'q10': player.field_maybe_none('first_q10') or '',
                'q11': player.field_maybe_none('first_q11') or '',
                'q12': player.field_maybe_none('first_q12') or '',
                'q13': player.field_maybe_none('first_q13') or '',
            }
        }

class Quiz(Page):
    form_model = 'player'
    form_fields = [
        'first_q1', 'first_q2', 'first_q3', 'first_q4', 'first_q5',
        'first_q6', 'first_q7', 'first_q8', 'first_q9', 'first_q10',
        'first_q11', 'first_q12', 'first_q13'
    ]

    def vars_for_template(player):
        return {
            'id': player.participant.label,
            'first_answers': {
                'q1': player.field_maybe_none('first_q1') or '',
                'q2': player.field_maybe_none('first_q2') or '',
                'q3': player.field_maybe_none('first_q3') or '',
                'q4': player.field_maybe_none('first_q4') or '',
                'q5': player.field_maybe_none('first_q5') or '',
                'q6': player.field_maybe_none('first_q6') or '',
                'q7': player.field_maybe_none('first_q7') or '',
                'q8': player.field_maybe_none('first_q8') or '',
                'q9': player.field_maybe_none('first_q9') or '',
                'q10': player.field_maybe_none('first_q10') or '',
                'q11': player.field_maybe_none('first_q11') or '',
                'q12': player.field_maybe_none('first_q12') or '',
                'q13': player.field_maybe_none('first_q13') or '',
            }
        }


page_sequence = [Quiz, Label]