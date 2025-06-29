from otree.api import *
import random
import json

class C(BaseConstants):
    NAME_IN_URL = 'marketpage2'
    PLAYERS_PER_GROUP = 4
    INITIAL_PRICE = 8
    PRICE_DROP = 2
    ASSET_VALUE = 20
    NUM_ROUNDS = 1
    GEOM_TIME_OUT = 0.125
    STATE = random.randint(0, 1) # 1 for good, 0 for bad
    PCORRECT = 0.7 # probability of correct signal
    PRIOR = 0.5 #initial prior belief
    TIMEOUT_SECONDS = random.randint(50,100)
    CHAT_TIMEOUT_SECONDS = 40

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    current_price = models.FloatField(initial = C.INITIAL_PRICE)
    prior = models.FloatField(initial=C.PRIOR)
    poll_count = models.IntegerField(initial=0)
    poll_players = models.LongStringField(initial='[]')
    current_signal = models.IntegerField(initial=-1)

class Player(BasePlayer):
    sold = models.BooleanField(initial=False)
    value = models.FloatField(initial=0)

class MarketWait(WaitPage):
    pass

class ChatWait(WaitPage):
    pass

class ResultsWait(WaitPage):
    pass

class Chat(Page):
    def get_timeout_seconds(player):
        return C.CHAT_TIMEOUT_SECONDS


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
            poll_players = json.loads(group.poll_players)
            if player.id_in_group not in poll_players:
                poll_players.append(player.id_in_group)
                group.poll_count += 1
                group.poll_players = json.dumps(poll_players)
            if group.poll_count < C.PLAYERS_PER_GROUP:
                return


            ## bayesian update
            # group prior is the belief that the asset is in good state
            if group.current_signal == -1:
                if C.STATE:
                    signal = 1 if random.random() < C.PCORRECT else 0
                    if signal:
                        new_prior = (C.PCORRECT * group.prior) / ((C.PCORRECT * group.prior) + ((1 - C.PCORRECT) * (1 - group.prior)))
                    else:
                        new_prior = ((1 - C.PCORRECT) * group.prior) / (((1 - C.PCORRECT) * group.prior) + (C.PCORRECT * (1 - group.prior)))
                else:
                    signal = 0 if random.random() < C.PCORRECT else 1
                    if signal:
                        new_prior = (C.PCORRECT * group.prior) / ((C.PCORRECT * group.prior) + ((1 - C.PCORRECT) * (1 - group.prior)))
                    else:
                        new_prior = ((1 - C.PCORRECT) * group.prior) / (((1 - C.PCORRECT) * group.prior) + (C.PCORRECT * (1 - group.prior)))
                group.prior = new_prior

        all_players = group.get_players()
        sold_status = {p.id_in_group: p.sold for p in all_players} 
        group.poll_count = 0
        group.poll_players = json.dumps([])
        group.current_signal = -1
        return {
            0: dict(
                new_price=group.current_price,
                old_price=group.current_price + C.PRICE_DROP,
                sold_status=sold_status,
                posterior=round(group.prior,2)*100,
            )  
        }


    def before_next_page(player, timeout_happened):
        if C.STATE and not player.sold:
            player.value = 2 * player.group.current_price
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

page_sequence = [ChatWait, Chat, MarketWait, Market, ResultsWait, Results]


