from otree.api import *
import random

class C(BaseConstants):
    NAME_IN_URL = 'marketpage2'
    PLAYERS_PER_GROUP = 8
    INITIAL_PRICE = 100
    PRICE_DROP = 10
    ASSET_VALUE = 90
    NUM_ROUNDS = 1
    GEOM_TIME_OUT = 0.125
    STATE = random.randint(0, 1) # 1 for good, 0 for bad
    PCORRECT = 0.7 # probability of correct signal
    PRIOR = 0.5 #initial prior belief
    TIMEOUT_SECONDS = random.randint(50,100)

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    current_price = models.FloatField(initial = C.INITIAL_PRICE)

class Player(BasePlayer):
    sold = models.BooleanField(initial=False)
    value = models.FloatField(initial=0)
    prior = models.FloatField(initial=C.PRIOR)

class MarketWait(WaitPage):
    pass

class Market(Page):
    @staticmethod
    def get_timeout_seconds(player):
        return C.TIMEOUT_SECONDS
    
    def vars_for_template(player):
        return{'initial_price': C.INITIAL_PRICE,
               'initial_prior': C.PRIOR*100}
    
    def live_method(player, data):
        group = player.group

        if data.get('action') == 'sell' and not player.sold:
            player.sold = True
            player.value = group.current_price
            group.current_price -= C.PRICE_DROP

        if data.get('action') == 'poll':
            ## bayesian update
            # player prior is the belief that the asset is in good state
            if C.STATE:
                signal = 1 if random.random() < C.PCORRECT else 0
                if signal:
                    new_prior = (C.PCORRECT * player.prior) / ((C.PCORRECT * player.prior) + ((1 - C.PCORRECT) * (1 - player.prior)))
                else:
                    new_prior = ((1 - C.PCORRECT) * player.prior) / (((1 - C.PCORRECT) * player.prior) + (C.PCORRECT * (1 - player.prior)))
            else:
                signal = 0 if random.random() < C.PCORRECT else 1
                if signal:
                    new_prior = (C.PCORRECT * player.prior) / ((C.PCORRECT * player.prior) + ((1 - C.PCORRECT) * (1 - player.prior)))
                else:
                    new_prior = ((1 - C.PCORRECT) * player.prior) / (((1 - C.PCORRECT) * player.prior) + (C.PCORRECT * (1 - player.prior)))
            player.prior = new_prior
        
        all_players = group.get_players()
        sold_status = {p.id_in_group: p.sold for p in all_players} 
        return {
            0: dict(
                new_price=group.current_price,
                old_price=group.current_price + C.PRICE_DROP,
                sold_status=sold_status,
                posterior=round(player.prior,2)*100,
            )  
        }

    def before_next_page(player, timeout_happened):
        if C.STATE and not player.sold:
            player.value = C.ASSET_VALUE
        else:
            # in bad state, assigning player values based on market price but there is random sequential ordering
            remaining_players = player.group.get_players()
            remaining_players = [p for p in remaining_players if not p.sold]
            min_price = player.group.current_price - (len(remaining_players) * C.PRICE_DROP)
            prices = list(range(int(player.group.current_price), int(min_price), -C.PRICE_DROP))
            random.shuffle(prices)
            for p, price in zip(remaining_players, prices):
                p.value = price
        return

class Results(Page):
    def vars_for_template(player):
        if C.STATE:
            st = 'good'
        else:
            st = 'bad'
        return {
            'value': player.value,
            'sold': player.sold,
            'state': st,
        }

page_sequence = [MarketWait, Market, Results]


