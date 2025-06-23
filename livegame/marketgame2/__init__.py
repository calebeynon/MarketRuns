from otree.api import *
import random
import matplotlib.pyplot as plt
import os
import uuid

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
    current_price = models.FloatField(initial=C.INITIAL_PRICE)
    prior = models.FloatField(initial=C.PRIOR)
    price_history = models.LongStringField(initial='[]')  # store as JSON string
    belief_history = models.LongStringField(initial='[]')

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
        group = player.group
        import json
        # Parse histories
        price_hist = json.loads(group.price_history) if group.price_history else [C.INITIAL_PRICE]
        belief_hist = json.loads(group.belief_history) if group.belief_history else [C.PRIOR*100]
        # Generate and save graphs
        static_dir = os.path.join('livegame', '_static', 'global')
        os.makedirs(static_dir, exist_ok=True)
        price_img = f'price_{uuid.uuid4().hex}.png'
        belief_img = f'belief_{uuid.uuid4().hex}.png'
        price_path = os.path.join(static_dir, price_img)
        belief_path = os.path.join(static_dir, belief_img)
        # Price graph
        plt.figure()
        plt.plot(price_hist, marker='o', color='red')
        plt.title('Market Price Over Time')
        plt.xlabel('Update Number')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(price_path)
        plt.close()
        # Belief graph
        plt.figure()
        plt.plot(belief_hist, marker='o', color='blue')
        plt.title('Belief of Good State Over Time')
        plt.xlabel('Update Number')
        plt.ylabel('Belief (%)')
        plt.tight_layout()
        plt.savefig(belief_path)
        plt.close()
        # URLs for template
        price_url = f'/static/global/{price_img}'
        belief_url = f'/static/global/{belief_img}'
        # Sold status
        all_players = group.get_players()
        sold_status = {p.id_in_group: p.sold for p in all_players}
        return {
            'initial_price': group.current_price,
            'initial_prior': group.prior*100,
            'price_img': price_url,
            'belief_img': belief_url,
            'sold_status': sold_status,
            'self_sold': player.sold,
        }

    @staticmethod
    def live_method(player, data):
        import json
        group = player.group
        # Load histories
        price_hist = json.loads(group.price_history) if group.price_history else [C.INITIAL_PRICE]
        belief_hist = json.loads(group.belief_history) if group.belief_history else [C.PRIOR*100]
        if data.get('action') == 'sell' and not player.sold:
            player.sold = True
            player.value = group.current_price
            group.current_price -= C.PRICE_DROP
        if data.get('action') == 'poll':
            # bayesian update
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
        # Update histories
        price_hist.append(group.current_price)
        belief_hist.append(group.prior*100)
        group.price_history = json.dumps(price_hist)
        group.belief_history = json.dumps(belief_hist)
        group.save()
        all_players = group.get_players()
        sold_status = {p.id_in_group: p.sold for p in all_players}
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


