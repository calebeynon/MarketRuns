from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'survey'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Player(BasePlayer):
    # Risk Aversion Questions - scale from 1 (very false) to 5 (very true)
    q1 = models.IntegerField(
        label="I don't like to take risks",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q2 = models.IntegerField(
        label="Compared to most people I know, I don't like to live life on the edge",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q3 = models.IntegerField(
        label="Compared to most people I know, I don't like to gamble on things",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q4 = models.IntegerField(
        label="I would rather be safe than sorry",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q5 = models.IntegerField(
        label="I have no desire to take unnecessary chances on things",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q6 = models.IntegerField(
        label="I avoid risky things",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Personality Trait Questions - Extraversion
    q7 = models.IntegerField(
        label="Is talkative",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q8 = models.IntegerField(
        label="Is full of energy",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q9 = models.IntegerField(
        label="Generates a lot of enthusiasm",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q10 = models.IntegerField(
        label="Is exciting",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q11 = models.IntegerField(
        label="Is outgoing, sociable",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Agreeableness
    q12 = models.IntegerField(
        label="Is helpful and unselfish with others",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q13 = models.IntegerField(
        label="Good relationship make with others",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q14 = models.IntegerField(
        label="Has a forgiving nature",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q15 = models.IntegerField(
        label="Is generally trusting",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q16 = models.IntegerField(
        label="Can be warm and close",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Conscientiousness
    q17 = models.IntegerField(
        label="Does a thorough job",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q18 = models.IntegerField(
        label="Can be careful",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q19 = models.IntegerField(
        label="Is a reliable and hardworking",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q20 = models.IntegerField(
        label="Perseveres until the task is finished",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q21 = models.IntegerField(
        label="Does things efficiently",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Neuroticism (from page 14)
    q22 = models.IntegerField(
        label="Is depressed, blue",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q23 = models.IntegerField(
        label="Uneasy, unable to cope with stress",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q24 = models.IntegerField(
        label="Can be tense",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q25 = models.IntegerField(
        label="Worries a lot",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q26 = models.IntegerField(
        label="Is emotionally fragile, easily upset",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Openness
    q27 = models.IntegerField(
        label="Is original, comes up with new ideas",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q28 = models.IntegerField(
        label="Is curious about many different things",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q29 = models.IntegerField(
        label="Resourceful",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q30 = models.IntegerField(
        label="Has an active imagination",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q31 = models.IntegerField(
        label="Is inventive",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Fear
    q32 = models.IntegerField(
        label="I am mostly concerned about the impression I've made in others",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q33 = models.IntegerField(
        label="I am afraid that others will not approve of me",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q34 = models.IntegerField(
        label="I am afraid that people will find fault with me",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q35 = models.IntegerField(
        label="I am concerned about others' views about me",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q36 = models.IntegerField(
        label="When I am talking to someone, I worry about what they may be thinking about me",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Anger
    q37 = models.IntegerField(
        label="I am a hot headed person",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q38 = models.IntegerField(
        label="I am quick tempered",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q39 = models.IntegerField(
        label="I suddenly get angry",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q40 = models.IntegerField(
        label="I fly off the handle",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q41 = models.IntegerField(
        label="When I get mad, I say nasty things",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Hope (from page 15)
    q42 = models.IntegerField(
        label="I follow my targets energetically",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q43 = models.IntegerField(
        label="My past experiences have prepared me well for my future",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q44 = models.IntegerField(
        label="I have been quite successful in life",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q45 = models.IntegerField(
        label="I fulfill my goals for myself",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    # Sadness
    q46 = models.IntegerField(
        label="If someone I'm talking with begins to cry, I get teary-eyed",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q47 = models.IntegerField(
        label="I get filled with sorrow when people talk about the death of their loved ones",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )
    
    q48 = models.IntegerField(
        label="I cry at sad movies",
        choices=[1, 2, 3, 4, 5],
        widget=widgets.RadioSelectHorizontal
    )

class Group(BaseGroup):
    pass

class Survey(Page):
    form_model = 'player'
    form_fields = [f'q{i}' for i in range(1, 49)]

page_sequence = [Survey]