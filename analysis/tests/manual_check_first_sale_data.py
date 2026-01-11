"""
Purpose: Human check of AI code to build first sale dataset, referring to build_first_sale_dataset.py and the corresponding tests authored by Claude Code
Author: Caleb Eynon
Date: 2025-01-11

"""

import pandas as pd
from pathlib import Path
import sys
import pytest
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import market_data as md

# =====
# File paths
# =====
DATASTORE = Path("/Users/caleb/Research/marketruns/datastore")
FIRST_SALE_DATA = DATASTORE / "derived" / "first_sale_data.csv"


# =====
# Load data
# =====
DATA = pd.read_csv(FIRST_SALE_DATA)
EXPERIMENT = md.parse_experiment(str(DATASTORE / "1_11-7-tr1/all_apps_wide_2025-11-07.csv"))
SESSION = EXPERIMENT.sessions[0]



# =====
# Main function
# =====
def main():
    signal_mismatches = 0
    seller_mismatches = 0
    period_mismatches = 0
    for segment_name,segment in SESSION.segments.items():
        for group_id,group in segment.groups.items():
            for round_num,round_obj in segment.rounds.items():
                c_sellers, c_signals, c_period = return_seg_group_sellers(segment_name,group_id,round_num)
                v_signal, v_period, v_n_sellers = round_signal_first_sale_data(segment_name,group_id,round_num)

                # Handle no-sales case
                if c_period is None and pd.isna(v_period):
                    continue

                if c_period != v_period: period_mismatches += 1
                if len(c_sellers) != v_n_sellers: seller_mismatches += 1
                all_signals = sum(1 for sig in c_signals if abs(sig - v_signal) > 1e-9)
                if all_signals != 0: signal_mismatches += 1
                
    assert(signal_mismatches == 0)
    assert(seller_mismatches == 0)
    assert(period_mismatches == 0)


def player_signal(period_obj,player_label) -> float:
    player = period_obj.get_player(player_label)
    return(player.signal)

def return_seg_group_sellers(seg,group,roun) -> Tuple[List[str], List[float], int]:
    segment = SESSION.get_segment(seg)
    group = segment.get_group(group)
    r = segment.get_round(roun)
    for period_num in sorted(r.periods.keys()):
        period = r.get_period(period_num)
        group_sellers = [p for p in group.player_labels if period.players.get(p) and period.players[p].sold_this_period] 
        seller_signals = [player_signal(period,p) for p in group_sellers]
        if group_sellers:
            return group_sellers, seller_signals, period_num
        elif period_num == max(sorted(r.periods.keys())):
            return [], [], None


def round_signal_first_sale_data(seg,group,roun) -> Tuple[float, int, int]:
    """ returns signal, period, n_sellers """
    matching_df = DATA[(DATA['segment'] == seg) & (DATA['round_num'] == roun) & (DATA['group_id'] == group)]
    return matching_df['signal_at_first_sale'].iloc[0], matching_df['first_sale_period'].iloc[0],matching_df['n_sellers_first_period'].iloc[0]



if __name__ == "__main__":
    main()

