from otree.api import *
import random

class C(BaseConstants):
    NAME_IN_URL = 'survey'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    coin_result = random.choice(["Heads", "Tails"])

class Subsession(BaseSubsession):
    pass

class Player(BasePlayer):
    # Risk Aversion Questions - scale from 1 (very false) to 5 (very true)
    q1 = models.StringField(
        label="calm",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q2 = models.StringField(
        label="relaxed",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q3 = models.StringField(
        label="content",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q4 = models.StringField(
        label="tense",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q5 = models.StringField(
        label="upset",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q6 = models.StringField(
        label="worried",
        choices=["Not at all", "Somewhat", "Moderately", "Very much"]
    )
    
    q7 = models.StringField(
        label="Extraverted, enthusiastic",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q8 = models.StringField(
        label="Critical, quarrelsome",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q9 = models.StringField(
        label="Dependable, self-disciplined",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q10 = models.StringField(
        label="Anxious, easily upset",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q11 = models.StringField(
        label="Open to new experiences, complex",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q12 = models.StringField(
        label="Reserved, quiet",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q13 = models.StringField(
        label="Sympathetic, warm",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q14 = models.StringField(
        label="Disorganized, careless",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q15 = models.StringField(
        label="Calm, emotionally stable",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q16 = models.StringField(
        label="Conventional, uncreative",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q17 = models.StringField(
        label="I plan tasks carefully",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q18 = models.StringField(
        label="I do things without thinking",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q19 = models.StringField(
        label="I don't pay attention",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q20 = models.StringField(
        label="I am self-controlled",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )   
    
    q21 = models.StringField(
        label="I concentrate easily",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q22 = models.StringField(
        label="I am a careful thinker",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q23 = models.StringField(
        label="I say things without thinking",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q24 = models.StringField(
        label="I act on the spur of the moment",
        choices=["Strongly Disagree", "Disagree Moderately", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree Moderately", "Strongly Agree"]
    )
    
    q25 = models.StringField(
        label="Your age", min = 16, max = 100
    )
    
    q26 = models.StringField(
        label="Your gender",
        choices=["Male", "Female", "Prefer not to say"]
    )
    
    # Openness
    q27 = models.StringField(
        label="Your classification",
        choices=["Freshman", "Sophomore", "Junior", "Senior"]
    )

    q28 = models.StringField(
        label="Your major"
    )
    
    allocate = models.IntegerField(initial=0)

    coin = models.StringField(choices=["Heads", "Tails"],widget=widgets.RadioSelectHorizontal)
    
class Group(BaseGroup):
    pass

class Survey(Page):
    form_model = 'player'
    form_fields = [f'q{i}' for i in range(1, 29)]

class Allocate(Page):
    form_model = 'player'
    form_fields = ['allocate', 'coin']
        
class Results(Page):
    def vars_for_template(player):
        allocate = player.allocate
        if player.coin == C.coin_result:
            player.payoff = (20 - allocate) + (allocate * 2.5)
        else:
            player.payoff = (20 - allocate)
        
        player.participant.vars['survey_payoff'] = float(player.payoff)
        
        return {
            'coin_result': C.coin_result,
            'payoff': float(player.payoff),
            'selected_coin': player.coin,
            'allocate': allocate
        }

page_sequence = [Survey, Allocate, Results]