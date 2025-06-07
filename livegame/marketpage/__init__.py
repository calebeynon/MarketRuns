from otree.api import *
import random

class C(BaseConstants):
    NAME_IN_URL = 'marketpage'
    PLAYERS_PER_GROUP = 8
    INITIAL_PRICE = 100
    PRICE_DROP = 10
    ASSET_VALUE = 90
    NUM_ROUNDS = 1
    TIME_OUT = 60

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    current_price = models.FloatField(initial = C.INITIAL_PRICE)

class Player(BasePlayer):
    sold = models.BooleanField(initial=False)
    value = models.FloatField(initial=0)



# PAGES
class MarketWait(WaitPage):
    pass

class Market(Page):
    live_method = 'market_live'
    timeout_seconds = C.TIME_OUT

    def vars_for_template(player):
        return {'current_price': player.group.current_price,
                'timeout_seconds': C.TIME_OUT,
                'sold': player.sold}

    @staticmethod
    def live_method(player, data):
        group = player.group

        if data.get('action') == 'sell' and not player.sold:
            player.sold = True
            player.value = group.current_price
            group.current_price -= C.PRICE_DROP
            group.save()


        all_players = group.get_players()
        sold_status = {p.id_in_group: p.sold for p in all_players} 
        return {
            0: dict(
                new_price=group.current_price,
                sold_status=sold_status,
            )
            
        }
    
    def before_next_page(player, timeout_happened):
        if not player.sold:
            player.value = random.choice([C.ASSET_VALUE,player.group.current_price])
        return 

class Results(Page):
    def vars_for_template(player):
        return {
            'value': player.value,
            'sold': player.sold,
        }
page_sequence = [MarketWait,Market,Results]