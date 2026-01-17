"""
Hierarchical data structure for oTree Market Runs experimental data.

This module provides classes to store and access market experimental data in a structured way
following the hierarchy: Session -> Segment -> Round -> Period -> Player.

Data Structure:
- Session: Container for all data from one experimental session
- Segment: A treatment segment (chat_noavg, chat_noavg2, etc.)
- Round: A trading round within a segment (1-14 rounds typical)
- Period: Individual trading periods within a round
- Player: Individual participant data for a specific period

Usage Example:
    exp = parse_experiment('/path/to/1_tr_data.csv')
    # Access player A's price in segment chat_noavg, round 1, period 1:
    price = exp.sessions[0].get_segment('chat_noavg').get_round(1).get_period(1).players['A'].price
    
    # Get all data as DataFrame:
    df = exp.as_dataframe(level='period')

Execute with: rye run python -m analysis.market_data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import re
from datetime import datetime
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
import argparse
import os

# Default path to data file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "datastore/raw/1_tr_data.csv"
CHAT_PATH = PROJECT_ROOT / "datastore/raw/1_tr_chat.csv"

class ParsingError(Exception):
    """Raised when there are issues parsing the experimental data."""
    pass


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message."""
    nickname: str  # Player label
    body: str
    timestamp: float
    participant_code: str
    id_in_session: int
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp)
    
    def __str__(self):
        return f"[{self.nickname}]: {self.body}"


@dataclass(frozen=True)
class PlayerPeriodData:
    """Individual player data for a specific period."""
    participant_id: int
    label: str  # A, B, C, etc.
    id_in_group: int
    sold: int  # Cumulative sold status (0 or 1)
    sold_this_period: bool  # Whether they sold in this specific period
    signal: Optional[float]
    price: Optional[float] 
    sell_click_time: Optional[float]  # Unix timestamp if sold this period
    state: int  # Market state
    payoff: Optional[float] = None  # Period payoff if available
    
    @property
    def sell_datetime(self) -> Optional[datetime]:
        """Convert sell_click_time to datetime if available."""
        if self.sell_click_time:
            return datetime.fromtimestamp(self.sell_click_time)
        return None
    
    def __str__(self):
        sold_str = "SOLD" if self.sold_this_period else "HOLD"
        price_str = f"@{self.price}" if self.price else ""
        return f"Player {self.label} ({sold_str}{price_str})"


@dataclass
class Period:
    """A single trading period within a round."""
    period_in_round: int
    players: Dict[str, PlayerPeriodData] = field(default_factory=dict)
    
    def add_player(self, player_data: PlayerPeriodData):
        """Add a player's data to this period."""
        self.players[player_data.label] = player_data
    
    def get_player(self, label: str) -> Optional[PlayerPeriodData]:
        """Get player data by label."""
        return self.players.get(label)
    
    @property
    def sellers(self) -> List[str]:
        """Get list of player labels who sold in this period."""
        return [label for label, player in self.players.items() if player.sold_this_period]
    
    @property
    def n_sellers(self) -> int:
        """Number of players who sold in this period."""
        return len(self.sellers)
    
    @property
    def avg_price(self) -> Optional[float]:
        """Average price among sellers in this period."""
        seller_prices = [p.price for p in self.players.values() 
                        if p.sold_this_period and p.price is not None]
        return np.mean(seller_prices) if seller_prices else None
    
    def __str__(self):
        return f"Period {self.period_in_round} ({self.n_sellers} sellers)"


@dataclass 
class Round:
    """A trading round containing multiple periods."""
    round_number_in_segment: int
    periods: OrderedDict[int, Period] = field(default_factory=OrderedDict)
    round_payoffs: Dict[str, float] = field(default_factory=dict)  # Final round payoffs by player
    chat_messages: List[ChatMessage] = field(default_factory=list)  # Pre-round chat messages
    
    def add_period(self, period: Period):
        """Add a period to this round."""
        self.periods[period.period_in_round] = period
    
    def get_period(self, period_num: int) -> Optional[Period]:
        """Get a specific period by number."""
        return self.periods.get(period_num)
    
    @property
    def last_period(self) -> Optional[Period]:
        """Get the last period of this round."""
        if self.periods:
            return list(self.periods.values())[-1]
        return None
    
    @property
    def n_periods(self) -> int:
        """Number of periods in this round."""
        return len(self.periods)
    
    @property
    def total_sellers(self) -> int:
        """Total number of unique players who sold at some point in this round."""
        all_sellers = set()
        for period in self.periods.values():
            all_sellers.update(period.sellers)
        return len(all_sellers)
    
    def get_player_across_periods(self, label: str) -> List[PlayerPeriodData]:
        """Get a player's data across all periods in this round."""
        player_data = []
        for period in self.periods.values():
            if label in period.players:
                player_data.append(period.players[label])
        return player_data
    
    def get_seller_period(self, label: str) -> Optional[int]:
        """Get the period number when a player sold, or None if they never sold."""
        for period_num, period in self.periods.items():
            if label in period.players and period.players[label].sold_this_period:
                return period_num
        return None
    
    def get_all_sellers_with_periods(self) -> Dict[str, int]:
        """Get a dictionary mapping player labels to the period they sold in."""
        sellers = {}
        for period_num, period in self.periods.items():
            for label, player in period.players.items():
                if player.sold_this_period and label not in sellers:
                    sellers[label] = period_num
        return sellers
    
    @property
    def sellers_by_period(self) -> Dict[int, List[str]]:
        """Get sellers grouped by the period they sold in."""
        by_period = {}
        for period_num, period in self.periods.items():
            sellers = period.sellers
            if sellers:
                by_period[period_num] = sellers
        return by_period
    
    def add_chat_message(self, message: ChatMessage):
        """Add a chat message to this round."""
        self.chat_messages.append(message)
    
    @property
    def n_chat_messages(self) -> int:
        """Number of chat messages in this round."""
        return len(self.chat_messages)
    
    def get_chat_by_player(self, label: str) -> List[ChatMessage]:
        """Get all chat messages from a specific player."""
        return [msg for msg in self.chat_messages if msg.nickname == label]
    
    def __str__(self):
        return f"Round {self.round_number_in_segment} ({self.n_periods} periods, {self.total_sellers} total sellers)"


@dataclass
class Group:
    """A group of players that remain together across rounds in a segment."""
    group_id: int
    player_labels: List[str] = field(default_factory=list)
    segment: Optional['Segment'] = field(default=None, repr=False)
    chat_channels: Dict[int, int] = field(default_factory=dict)  # round_number -> channel_number
    
    def add_player(self, label: str):
        """Add a player to this group."""
        if label not in self.player_labels:
            self.player_labels.append(label)
    
    @property
    def size(self) -> int:
        """Number of players in the group."""
        return len(self.player_labels)
    
    def get_players_in_period(self, round_number: int, period_number: int) -> Dict[str, PlayerPeriodData]:
        """Get all player objects for this group in a specific period."""
        if not self.segment:
            return {}
        
        round_obj = self.segment.get_round(round_number)
        if not round_obj:
            return {}
        
        period = round_obj.get_period(period_number)
        if not period:
            return {}
        
        return {label: period.players[label] for label in self.player_labels if label in period.players}
    
    def get_players_in_round(self, round_number: int) -> Dict[str, List[PlayerPeriodData]]:
        """Get all player objects for this group across all periods in a round."""
        if not self.segment:
            return {}
        
        round_obj = self.segment.get_round(round_number)
        if not round_obj:
            return {}
        
        player_data = {}
        for label in self.player_labels:
            player_data[label] = round_obj.get_player_across_periods(label)
        return player_data
    
    def get_players_across_segment(self) -> Dict[str, Dict[int, List[PlayerPeriodData]]]:
        """Get all player objects for this group across all rounds in the segment."""
        if not self.segment:
            return {}
        
        player_data = {}
        for label in self.player_labels:
            player_data[label] = self.segment.get_player_across_rounds(label)
        return player_data
    
    def get_chat_for_round(self, round_number: int) -> List[ChatMessage]:
        """Get all chat messages for this group in a specific round."""
        if not self.segment:
            return []
        
        round_obj = self.segment.get_round(round_number)
        if not round_obj:
            return []
        
        return round_obj.chat_messages
    
    def get_chat_across_segment(self) -> Dict[int, List[ChatMessage]]:
        """Get all chat messages for this group across all rounds."""
        if not self.segment:
            return {}
        
        chat_data = {}
        for round_num, round_obj in self.segment.rounds.items():
            if round_obj.chat_messages:
                chat_data[round_num] = round_obj.chat_messages
        return chat_data
    
    def __str__(self):
        return f"Group {self.group_id} ({self.size} players: {', '.join(sorted(self.player_labels))})"


@dataclass
class Segment:
    """A treatment segment containing multiple rounds."""
    name: str
    rounds: OrderedDict[int, Round] = field(default_factory=OrderedDict)
    groups: Dict[int, Group] = field(default_factory=dict)
    
    def add_round(self, round_obj: Round):
        """Add a round to this segment."""
        self.rounds[round_obj.round_number_in_segment] = round_obj
    
    def get_round(self, round_number: int) -> Optional[Round]:
        """Get a specific round by number."""
        return self.rounds.get(round_number)
    
    def add_group(self, group: Group):
        """Add a group to this segment."""
        self.groups[group.group_id] = group
    
    def get_group(self, group_id: int) -> Optional[Group]:
        """Get a specific group by ID."""
        return self.groups.get(group_id)
    
    def get_group_by_player(self, label: str) -> Optional[Group]:
        """Find which group a player belongs to."""
        for group in self.groups.values():
            if label in group.player_labels:
                return group
        return None
    
    @property
    def n_rounds(self) -> int:
        """Number of rounds in this segment."""
        return len(self.rounds)
    
    @property
    def n_groups(self) -> int:
        """Number of groups in this segment."""
        return len(self.groups)
    
    def get_player_across_rounds(self, label: str) -> Dict[int, List[PlayerPeriodData]]:
        """Get a player's data across all rounds in this segment."""
        player_data = {}
        for round_num, round_obj in self.rounds.items():
            player_data[round_num] = round_obj.get_player_across_periods(label)
        return player_data
    
    def __str__(self):
        return f"Segment '{self.name}' ({self.n_rounds} rounds, {self.n_groups} groups)"


@dataclass
class Session:
    """Complete session data containing all segments."""
    session_code: str
    segments: Dict[str, Segment] = field(default_factory=dict)
    participant_labels: Dict[int, str] = field(default_factory=dict)  # participant_id -> label
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_segment(self, segment: Segment):
        """Add a segment to this session."""
        self.segments[segment.name] = segment
    
    def get_segment(self, name: str) -> Optional[Segment]:
        """Get a segment by name."""
        return self.segments.get(name)
    
    @property
    def segments_by_name(self) -> List[str]:
        """Get a dictionary of segments by name."""
        return [segment.name for segment in self.segments.values()]

    @property
    def n_participants(self) -> int:
        """Number of participants in this session."""
        return len(self.participant_labels)
    
    def get_player_across_session(self, label: str) -> Dict[str, Dict[int, List[PlayerPeriodData]]]:
        """Get a player's data across all segments and rounds."""
        player_data = {}
        for segment_name, segment in self.segments.items():
            player_data[segment_name] = segment.get_player_across_rounds(label)
        return player_data
    
    def __str__(self):
        return f"Session {self.session_code} ({len(self.segments)} segments, {self.n_participants} participants)"


@dataclass
class MarketRunsExperiment:
    """Experiment-level container holding multiple sessions."""
    name: Optional[str] = None
    sessions: List[Session] = field(default_factory=list)
    
    def add_session(self, session: Session):
        """Add a session to the experiment."""
        self.sessions.append(session)
    
    def get_session(self, session_code: str) -> Optional[Session]:
        """Get a session by code."""
        for session in self.sessions:
            if session.session_code == session_code:
                return session
        return None
    
    @property
    def sessions_by_code(self) -> List[str]:
        """Get a list of sessions by code."""
        return [session.session_code for session in self.sessions]
    
    @property
    def n_sessions(self) -> int:
        """Number of sessions in the experiment."""
        return len(self.sessions)
    
    @property
    def total_participants(self) -> int:
        """Total participants across all sessions."""
        return sum(session.n_participants for session in self.sessions)
    
    def as_dataframe(self, level: str = 'period') -> pd.DataFrame:
        """
        Convert experiment data to a flat DataFrame.
        
        Args:
            level: 'period', 'round', or 'player' - level of aggregation
            
        Returns:
            DataFrame with hierarchical data flattened
        """
        records = []
        
        for session in self.sessions:
            for segment_name, segment in session.segments.items():
                for round_num, round_obj in segment.rounds.items():
                    
                    if level == 'round':
                        # One row per player per round
                        for label in session.participant_labels.values():
                            if round_obj.periods:
                                # Use last period for round-level data
                                last_period = round_obj.last_period
                                if label in last_period.players:
                                    player = last_period.players[label]
                                    records.append({
                                        'session_code': session.session_code,
                                        'segment': segment_name,
                                        'round': round_num,
                                        'label': label,
                                        'participant_id': player.participant_id,
                                        'id_in_group': player.id_in_group,
                                        'final_sold_status': player.sold,
                                    'round_payoff': round_obj.round_payoffs.get(label),
                                    'total_sellers_in_round': round_obj.total_sellers,
                                    'n_periods': round_obj.n_periods,
                                    'group_id': segment.get_group_by_player(label).group_id if segment.get_group_by_player(label) else None
                                })
                    
                    else:  # period level (default)
                        for period_num, period in round_obj.periods.items():
                            for label, player in period.players.items():
                                records.append({
                                    'session_code': session.session_code,
                                    'segment': segment_name,
                                    'round': round_num,
                                    'period': period_num,
                                    'label': label,
                                    'participant_id': player.participant_id,
                                    'id_in_group': player.id_in_group,
                                    'sold': player.sold,
                                    'sold_this_period': player.sold_this_period,
                                    'signal': player.signal,
                                    'price': player.price,
                                    'sell_click_time': player.sell_click_time,
                                    'state': player.state,
                                    'payoff': player.payoff,
                                    'round_payoff': round_obj.round_payoffs.get(label),
                                    'group_id': segment.get_group_by_player(label).group_id if segment.get_group_by_player(label) else None
                                })
        
        return pd.DataFrame(records) if records else pd.DataFrame()
    
    def __str__(self):
        return f"MarketRunsExperiment '{self.name}' ({self.n_sessions} sessions, {self.total_participants} total participants)"


def parse_chat_data(chat_path: str, experiment: MarketRunsExperiment) -> None:
    """
    Parse chat data from CSV file and add to existing experiment structure.
    
    Args:
        chat_path: Path to the chat CSV file
        experiment: MarketRunsExperiment object to add chat data to
    """
    print(f"\nLoading chat data from: {chat_path}")
    
    if not os.path.exists(chat_path):
        print(f"Warning: Chat file not found: {chat_path}")
        return
    
    # Read chat CSV file
    chat_df = pd.read_csv(chat_path)
    print(f"Loaded {len(chat_df)} chat messages")
    
    # Parse channel format: 1-{segment}-{channel_number}
    channel_pattern = re.compile(r'^1-([^-]+)-(\d+)$')
    
    # Group chat messages by session and segment
    for session in experiment.sessions:
        session_chat = chat_df[chat_df['session_code'] == session.session_code]
        
        if session_chat.empty:
            continue
        
        for segment_name, segment in session.segments.items():
            # Filter chat for this segment
            segment_chat = session_chat[session_chat['channel'].str.contains(f'1-{segment_name}-', na=False)]
            
            if segment_chat.empty:
                continue
            
            print(f"  Processing chat for {session.session_code}/{segment_name}: {len(segment_chat)} messages")
            
            # Build mapping: channel_number -> group_id by matching player labels
            channel_to_group = {}
            
            for _, row in segment_chat.iterrows():
                match = channel_pattern.match(row['channel'])
                if not match:
                    continue
                
                seg_name = match.group(1)
                channel_num = int(match.group(2))
                nickname = row['nickname']
                
                # Find which group this player belongs to
                if channel_num not in channel_to_group:
                    for group_id, group in segment.groups.items():
                        if nickname in group.player_labels:
                            channel_to_group[channel_num] = group_id
                            break
            
            # Determine round boundaries (4 channels per round)
            if not channel_to_group:
                continue
            
            min_channel = min(channel_to_group.keys())
            max_channel = max(channel_to_group.keys())
            channels_per_round = 4
            
            # Map channels to rounds
            channel_to_round = {}
            for channel_num in range(min_channel, max_channel + 1):
                round_num = ((channel_num - min_channel) // channels_per_round) + 1
                channel_to_round[channel_num] = round_num
            
            # Store channel mappings in groups
            for channel_num, group_id in channel_to_group.items():
                round_num = channel_to_round[channel_num]
                if group_id in segment.groups:
                    segment.groups[group_id].chat_channels[round_num] = channel_num
            
            # Add chat messages to appropriate rounds
            for _, row in segment_chat.iterrows():
                match = channel_pattern.match(row['channel'])
                if not match:
                    continue
                
                channel_num = int(match.group(2))
                
                if channel_num not in channel_to_group or channel_num not in channel_to_round:
                    continue
                
                group_id = channel_to_group[channel_num]
                round_num = channel_to_round[channel_num]
                
                # Create chat message
                message = ChatMessage(
                    nickname=row['nickname'],
                    body=row['body'],
                    timestamp=float(row['timestamp']),
                    participant_code=row['participant_code'],
                    id_in_session=int(row['id_in_session'])
                )
                
                # Add to appropriate round
                if round_num in segment.rounds:
                    segment.rounds[round_num].add_chat_message(message)
    
    print(f"Chat data parsing complete!")


def parse_experiment(csv_path: str, chat_path: Optional[str] = None) -> MarketRunsExperiment:
    """
    Parse experimental data from CSV file into hierarchical structure.
    
    Args:
        csv_path: Path to the CSV file containing experimental data
        chat_path: Optional path to the chat CSV file
        
    Returns:
        MarketRunsExperiment object containing all experimental data
    """
    print(f"Loading experimental data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise ParsingError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows of data")
    
    # Detect segment names from column patterns
    segment_pattern = re.compile(r'^(chat_noavg\d*)\.\d+\.player\.')
    segment_names = set()
    for col in df.columns:
        match = segment_pattern.match(col)
        if match:
            segment_name = match.group(1)
            segment_names.add(segment_name)
    
    segment_names = sorted(segment_names)
    print(f"Detected segments: {segment_names}")
    
    experiment = MarketRunsExperiment(name="Market Runs Experiment")
    
    # Process each unique session
    session_codes = df['session.code'].unique()
    print(f"Processing {len(session_codes)} sessions...")
    
    for session_code in session_codes:
        print(f"  Processing session: {session_code}")
        session_df = df[df['session.code'] == session_code].copy()
        
        session = Session(session_code=session_code)
        
        # Store session metadata
        if not session_df.empty:
            first_row = session_df.iloc[0]
            session.metadata.update({
                'participation_fee': first_row.get('session.config.participation_fee'),
                'real_world_currency_per_point': first_row.get('session.config.real_world_currency_per_point'),
                'room': first_row.get('session.config.room'),
                'is_demo': first_row.get('session.is_demo', False)
            })
        
        # Build participant label mapping
        for _, row in session_df.iterrows():
            if pd.notna(row['participant.label']):
                session.participant_labels[row['participant.id_in_session']] = row['participant.label']
        
        # Process each segment
        for segment_name in segment_names:
            segment = Segment(name=segment_name)
            
            # Find columns for this segment
            segment_cols = [col for col in df.columns if col.startswith(f'{segment_name}.')]
            if not segment_cols:
                continue
                
            print(f"    Processing segment: {segment_name} ({len(segment_cols)} columns)")
            
            # Extract period numbers from column names
            period_pattern = re.compile(f'^{re.escape(segment_name)}\\.(\\d+)\\.player\\.')
            period_numbers = set()
            for col in segment_cols:
                match = period_pattern.match(col)
                if match:
                    period_numbers.add(int(match.group(1)))
            
            if not period_numbers:
                continue
                
            period_numbers = sorted(period_numbers)
            
            # Track groups for this segment (groups remain constant across rounds)
            segment_groups = {}  # group_id -> set of player labels
            
            # Process each participant row
            for _, row in session_df.iterrows():
                participant_id = row['participant.id_in_session']
                label = row['participant.label']
                
                if pd.isna(label):
                    continue
                
                # Track sold status per round (resets each round)
                round_sold_status = {}  # round_num -> sold_status
                
                # Process each period for this participant
                for period_num in period_numbers:
                    # Extract data for this period
                    period_prefix = f'{segment_name}.{period_num}.player.'
                    
                    # Check if this participant has data for this period
                    id_col = f'{period_prefix}id_in_group'
                    if id_col not in row or pd.isna(row[id_col]):
                        continue
                    
                    # Extract group information (constant across rounds in a segment)
                    group_col = f'{segment_name}.{period_num}.group.id_in_subsession'
                    if group_col in row and pd.notna(row[group_col]):
                        group_id = int(row[group_col])
                        if group_id not in segment_groups:
                            segment_groups[group_id] = set()
                        segment_groups[group_id].add(label)
                    
                    # Get round and period info first
                    round_num = row.get(f'{period_prefix}round_number_in_segment', 1)
                    period_in_round = row.get(f'{period_prefix}period_in_round', 1)
                    
                    if pd.isna(round_num):
                        round_num = 1
                    if pd.isna(period_in_round):
                        period_in_round = 1
                        
                    round_num = int(round_num)
                    period_in_round = int(period_in_round)
                    
                    # Extract player data
                    sold_value = row.get(f'{period_prefix}sold', 0)
                    if pd.isna(sold_value):
                        sold_value = 0
                    sold_value = int(sold_value)
                    
                    sell_click_time = row.get(f'{period_prefix}sell_click_time')
                    if pd.isna(sell_click_time):
                        sell_click_time = None
                    
                    # Initialize round sold status if needed
                    if round_num not in round_sold_status:
                        round_sold_status[round_num] = 0
                    
                    # Determine if sold this period (has sell_click_time or sold value increased from previous period in round)
                    sold_this_period = pd.notna(sell_click_time) or (sold_value > round_sold_status[round_num])
                    round_sold_status[round_num] = max(round_sold_status[round_num], sold_value)
                    
                    # Create player data
                    player_data = PlayerPeriodData(
                        participant_id=participant_id,
                        label=label,
                        id_in_group=int(row[id_col]),
                        sold=sold_value,  # Use the actual sold value from CSV (per-round)
                        sold_this_period=sold_this_period,
                        signal=row.get(f'{period_prefix}signal'),
                        price=row.get(f'{period_prefix}price'),
                        sell_click_time=sell_click_time,
                        state=int(row.get(f'{period_prefix}state', 0)),
                        payoff=row.get(f'{period_prefix}payoff')
                    )
                    
                    # Add to appropriate round and period
                    if round_num not in segment.rounds:
                        segment.rounds[round_num] = Round(round_number_in_segment=round_num)
                    
                    round_obj = segment.rounds[round_num]
                    
                    if period_in_round not in round_obj.periods:
                        round_obj.periods[period_in_round] = Period(period_in_round=period_in_round)
                    
                    period_obj = round_obj.periods[period_in_round]
                    period_obj.add_player(player_data)
                
                # Extract round payoffs (from round_X_payoff columns)
                for round_num in range(1, 15):  # Assuming up to 14 rounds
                    payoff_col = f'{segment_name}.1.player.round_{round_num}_payoff'  # Payoffs are in first period
                    if payoff_col in row and not pd.isna(row[payoff_col]) and round_num in segment.rounds:
                        segment.rounds[round_num].round_payoffs[label] = float(row[payoff_col])
            
            # Create Group objects from collected group data
            for group_id, player_labels in segment_groups.items():
                group = Group(group_id=group_id, player_labels=sorted(list(player_labels)), segment=segment)
                segment.add_group(group)
            
            if segment.rounds:
                session.add_segment(segment)
        
        if session.segments:
            experiment.add_session(session)
    
    print(f"Parsing complete! Loaded {experiment.n_sessions} sessions with {experiment.total_participants} total participants.")
    
    # Parse chat data if provided
    if chat_path:
        parse_chat_data(chat_path, experiment)
    
    return experiment


def main():
    """Main function to load and analyze experimental data."""
    parser = argparse.ArgumentParser(description='Load and analyze Market Runs experimental data')
    parser.add_argument('--csv-path', default=CSV_PATH, help='Path to CSV data file')
    parser.add_argument('--chat-path', default=CHAT_PATH, help='Path to chat CSV file')
    parser.add_argument('--export', help='Export to parquet file (specify path)')
    parser.add_argument('--summary', action='store_true', help='Print detailed summary')
    
    args = parser.parse_args()
    
    try:
        # Load experimental data
        experiment = parse_experiment(args.csv_path, args.chat_path)
        
        # Print basic summary
        print("\n" + "="*60)
        print("EXPERIMENTAL DATA SUMMARY")
        print("="*60)
        print(f"Experiment: {experiment}")
        
        for i, session in enumerate(experiment.sessions):
            print(f"\nSession {i+1}: {session}")
            for segment_name, segment in session.segments.items():
                print(f"  {segment}")
                
                if args.summary:
                    for round_num, round_obj in list(segment.rounds.items())[:3]:  # Show first 3 rounds
                        print(f"    {round_obj}")
                        for period_num, period in list(round_obj.periods.items())[:2]:  # Show first 2 periods
                            print(f"      {period}")
                    if len(segment.rounds) > 3:
                        print(f"    ... and {len(segment.rounds)-3} more rounds")
        
        # Export if requested
        if args.export:
            print(f"\nExporting data to: {args.export}")
            df = experiment.as_dataframe(level='period')
            df.to_parquet(args.export, index=False)
            print(f"Exported {len(df)} rows to parquet file.")
        
        # Show sample access pattern
        if experiment.sessions:
            session = experiment.sessions[0]
            segment_name = list(session.segments.keys())[0]
            segment = session.get_segment(segment_name)
            if segment and segment.rounds:
                round_obj = segment.get_round(1)
                if round_obj and round_obj.periods:
                    period = round_obj.get_period(1)
                    if period and period.players:
                        player_label = list(period.players.keys())[0]
                        player = period.get_player(player_label)
                        print(f"\nSample access:")
                        print(f"experiment.sessions[0].get_segment('{segment_name}').get_round(1).get_period(1).get_player('{player_label}')")
                        print(f"Result: {player}")
                        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return experiment


if __name__ == '__main__':
    main()