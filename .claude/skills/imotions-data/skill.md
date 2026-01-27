# iMotions Facial Expression Data Skill

Use this skill when working with iMotions facial expression data from the MarketRuns experiment.

## When to Use

- Processing or analyzing facial expression data from `datastore/imotions/`
- Extracting emotion metrics during specific experiment phases
- Matching facial data to oTree sell events
- Building derived datasets combining iMotions and oTree data

## Data Location

```
datastore/imotions/
├── 1/                    # Session 1 (16 participants)
├── 2/                    # Session 2 (16 participants + ExportMerge.csv to ignore)
├── 3/                    # Session 3 (16 participants + ExportMerge.csv to ignore)
├── 4/                    # Session 4 (16 participants)
├── 5/                    # Session 5 (16 participants)
└── 6/                    # Session 6 (16 participants)
```

## File Structure

### Metadata Rows (1-24)

Skip these rows when reading data. Key metadata:
- Row 3: `Respondent Name` contains participant letter ID (e.g., "R3")
- Row 9: `Recording time` contains recording start datetime

### Data Header (Row 25)

Column names for data. Use `skiprows=24` when reading with pandas.

### Key Properties

| Property | Value |
|----------|-------|
| Sampling rate | ~25 Hz |
| Duration | ~70 minutes |
| Rows per file | ~100,000 |

## Column Reference

### Emotion Columns (Index 10-21)

```python
EMOTION_COLS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Joy',
    'Sadness', 'Surprise', 'Engagement', 'Valence',
    'Sentimentality', 'Confusion', 'Neutral'
]
```

### Action Unit Columns (Index 22-44)

```python
ACTION_UNIT_COLS = [
    'Attention', 'Brow Furrow', 'Brow Raise', 'Cheek Raise',
    'Chin Raise', 'Dimpler', 'Eye Closure', 'Eye Widen',
    'Inner Brow Raise', 'Jaw Drop', 'Lip Corner Depressor',
    'Lip Press', 'Lip Pucker', 'Lip Stretch', 'Lip Suck',
    'Lid Tighten', 'Mouth Open', 'Nose Wrinkle', 'Smile',
    'Smirk', 'Upper Lip Raise', 'Blink', 'BlinkRate'
]
```

### Head Rotation Columns (Index 45-48)

```python
HEAD_ROTATION_COLS = ['Pitch', 'Yaw', 'Roll', 'Interocular Distance']
```

### Annotation Column (Index 49)

Column name: `Respondent Annotations active`

## Annotation Encoding

Pattern: `s{segment}r{round}m{period}{phase}`

### Segment Mapping

| Annotation | oTree App | Has Chat |
|------------|-----------|----------|
| s1 | chat_noavg | No |
| s2 | chat_noavg2 | No |
| s3 | chat_noavg3 | Yes |
| s4 | chat_noavg4 | Yes |

### Phase Types

| Phase | Description |
|-------|-------------|
| `MarketPeriod` | **Active trading** - primary phase for analysis |
| `MarketPeriodWait` | Waiting between periods |
| `MarketPeriodPayoffWait` | After selling, waiting for payoff |
| `ResultsWait` | Before results display |
| `Results` | End of round results |
| `Chat` | Chat period (s3, s4 only) |
| `ChatWait` | After chat |
| `SegmentIntro` | Segment start |
| `NewRule` | Rule change introduction |

### Special Annotations

- `Label` - Calibration phase
- `Allocate` - Asset allocation
- `Survey` - Post-experiment survey
- Empty - Pre-experiment or transitions

## Code Templates

### Loading Data

```python
import pandas as pd
from pathlib import Path

def load_imotions(session: int, participant_letter: str) -> pd.DataFrame:
    """Load iMotions data for a participant.

    Args:
        session: Session number (1-6)
        participant_letter: Single letter ID (A-R, excluding I, O)

    Returns:
        DataFrame with facial expression data
    """
    suffix = session + 2  # Session 1 = suffix 3

    base_path = Path('datastore/imotions') / str(session)
    files = list(base_path.glob(f'*_{participant_letter}{suffix}.csv'))

    if not files:
        raise FileNotFoundError(f"No file for {participant_letter} in session {session}")

    return pd.read_csv(files[0], skiprows=24, encoding='utf-8-sig')
```

### Filtering by Phase

```python
def filter_by_phase(df: pd.DataFrame, segment: int, round_num: int,
                    period: int, phase: str = 'MarketPeriod') -> pd.DataFrame:
    """Filter to specific experiment phase.

    Args:
        df: iMotions DataFrame
        segment: Segment number (1-4)
        round_num: Round number (1-14)
        period: Period number within round
        phase: Phase name (default: 'MarketPeriod')

    Returns:
        Filtered DataFrame
    """
    annotation = f's{segment}r{round_num}m{period}{phase}'
    return df[df['Respondent Annotations active'] == annotation]
```

### Extracting Emotions During Market Period

```python
def get_market_emotions(df: pd.DataFrame, segment: int, round_num: int,
                        period: int) -> pd.DataFrame:
    """Get emotion metrics during a market period.

    Returns DataFrame with Timestamp and emotion columns.
    """
    filtered = filter_by_phase(df, segment, round_num, period, 'MarketPeriod')

    emotion_cols = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy',
                    'Sadness', 'Surprise', 'Engagement', 'Valence']

    return filtered[['Timestamp'] + emotion_cols].copy()
```

### Aggregating Emotions for Analysis

```python
def aggregate_period_emotions(df: pd.DataFrame, segment: int, round_num: int,
                              period: int) -> dict:
    """Compute summary statistics for emotions during a market period.

    Returns dict with mean, std, min, max for each emotion.
    """
    emotions = get_market_emotions(df, segment, round_num, period)

    if emotions.empty:
        return None

    result = {}
    for col in emotions.columns[1:]:  # Skip Timestamp
        result[f'{col}_mean'] = emotions[col].mean()
        result[f'{col}_std'] = emotions[col].std()
        result[f'{col}_max'] = emotions[col].max()

    return result
```

## Participant ID Mapping

### File Pattern

`{order}_{letter}{suffix}.csv` where:
- `order`: 001-016
- `letter`: Participant letter (A-R, excluding I, O)
- `suffix`: Session + 2 (session 1 = 3, session 2 = 4, etc.)

### To oTree Mapping

Match the letter ID to `participant.label` in oTree data for the same session.

## Data Quality Notes

### Filtering Recommendations

1. **Attention < 50**: Unreliable face detection
2. **Empty annotations**: Skip pre-experiment rows
3. **Timestamp gaps > 100ms**: Potential recording issues

### Common Issues

- **NaN values**: Face not detected (blinks, looking away)
- **ExportMerge.csv**: Ignore these files in sessions 2 and 3
- **First ~20 min**: May contain calibration/labeling before experiment

## Integration with oTree

### Matching to Sell Events

1. Get sell timestamp from oTree `PlayerPeriodData.sold_time` (if available)
2. Find `MarketPeriod` annotation for matching segment/round/period
3. Extract facial data within that time window
4. Compute pre-sell and post-sell emotion metrics

### Cross-Reference

```python
# Example: Get emotions before and after a sell decision
def get_sell_emotions(imotions_df, segment, round_num, period,
                      sell_timestamp_ms, window_ms=5000):
    """Get emotions around a sell event.

    Args:
        sell_timestamp_ms: Sell time in ms (relative to iMotions start)
        window_ms: Time window before/after sell
    """
    market_data = filter_by_phase(imotions_df, segment, round_num,
                                  period, 'MarketPeriod')

    pre_sell = market_data[
        (market_data['Timestamp'] >= sell_timestamp_ms - window_ms) &
        (market_data['Timestamp'] < sell_timestamp_ms)
    ]

    post_sell = market_data[
        (market_data['Timestamp'] >= sell_timestamp_ms) &
        (market_data['Timestamp'] < sell_timestamp_ms + window_ms)
    ]

    return pre_sell, post_sell
```

## Reference

Full documentation: `issues/issue_15_imotions_documentation.md`
