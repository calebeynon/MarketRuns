# Issue #15: Document iMotions Facial Expression Data Processing Pipeline

## Overview

This document provides comprehensive documentation for working with iMotions facial expression data collected during the MarketRuns experiments. The data uses Affectiva AFFDEX facial analysis to capture emotions, action units, and head movements throughout each experimental session.

## Data Location

```
datastore/imotions/
├── 1/                    # Session 1 (Treatment 1)
│   ├── 001_R3.csv       # Participant R3
│   ├── 002_Q3.csv       # Participant Q3
│   └── ...              # 16 participants total
├── 2/                    # Session 2 (Treatment 2)
├── 3/                    # Session 3
├── 4/                    # Session 4
├── 5/                    # Session 5
└── 6/                    # Session 6
```

**Note:** Sessions 2 and 3 contain an `ExportMerge.csv` file that should be ignored. Each session has exactly 16 participants.

## File Structure

### Metadata Header (Rows 1-24)

The first 24 rows contain metadata in CSV format:

| Row | Content | Example |
|-----|---------|---------|
| 2 | Study name | `CPMR_1` |
| 3 | Respondent Name | `R3` with UUID |
| 4 | Respondent Age | `0` (not collected) |
| 5 | Respondent Gender | `MALE` or `FEMALE` |
| 6 | Respondent Group | `Default` |
| 7 | iMotions version | `10.0.37173.5` |
| 9 | Recording time | Date and time with timezone |
| 10 | Export time | Date and time of export |
| 11 | Sensor info | `Affectiva AFFDEX, Manufacturer: Affectiva` |

### Data Header (Row 25)

Row 25 contains column names. Data rows start at row 26.

### Data Properties

| Property | Value |
|----------|-------|
| Sampling rate | ~25 Hz |
| Duration | ~70 minutes per participant |
| Rows per file | ~100,000 data rows |
| File size | 40-90 MB per participant |

## Column Reference

### Core Columns (0-9)

| Index | Column | Description |
|-------|--------|-------------|
| 0 | Row | Row number (1-indexed) |
| 1 | Timestamp | Milliseconds since recording start |
| 2 | EventSource | Event source identifier |
| 3 | SlideEvent | Slide event type (StartSlide, StartMedia) |
| 4 | StimType | Stimulus type (TestImage) |
| 5 | Duration | Stimulus duration in ms |
| 6 | CollectionPhase | Collection phase |
| 7 | SourceStimuliName | Stimulus name |
| 8 | EventSource | AFFDEX event source |
| 9 | SampleNumber | Sample counter |

### Emotion Columns (10-21)

| Index | Column | Range | Description |
|-------|--------|-------|-------------|
| 10 | Anger | 0-100 | Evidence of anger |
| 11 | Contempt | 0-100 | Evidence of contempt |
| 12 | Disgust | 0-100 | Evidence of disgust |
| 13 | Fear | 0-100 | Evidence of fear |
| 14 | Joy | 0-100 | Evidence of joy |
| 15 | Sadness | 0-100 | Evidence of sadness |
| 16 | Surprise | 0-100 | Evidence of surprise |
| 17 | Engagement | 0-100 | Facial muscle activation measure |
| 18 | Valence | -100 to 100 | Positive/negative experience |
| 19 | Sentimentality | 0-100 | Evidence of sentimentality |
| 20 | Confusion | 0-100 | Evidence of confusion |
| 21 | Neutral | 0-100 | Evidence of neutral expression |

### Action Unit Columns (22-44)

| Index | Column | Range | Description |
|-------|--------|-------|-------------|
| 22 | Attention | 0-100 | Focus based on head orientation |
| 23 | Brow Furrow | 0-100 | Brow furrow intensity |
| 24 | Brow Raise | 0-100 | Brow raise intensity |
| 25 | Cheek Raise | 0-100 | Cheek raise intensity |
| 26 | Chin Raise | 0-100 | Chin raise intensity |
| 27 | Dimpler | 0-100 | Dimple intensity |
| 28 | Eye Closure | 0-100 | Eye closure intensity |
| 29 | Eye Widen | 0-100 | Eye widening intensity |
| 30 | Inner Brow Raise | 0-100 | Inner brow raise intensity |
| 31 | Jaw Drop | 0-100 | Jaw drop intensity |
| 32 | Lip Corner Depressor | 0-100 | Lip corner depression |
| 33 | Lip Press | 0-100 | Lip pressing intensity |
| 34 | Lip Pucker | 0-100 | Lip puckering intensity |
| 35 | Lip Stretch | 0-100 | Lip stretching intensity |
| 36 | Lip Suck | 0-100 | Lip sucking intensity |
| 37 | Lid Tighten | 0-100 | Lid tightening intensity |
| 38 | Mouth Open | 0-100 | Mouth opening intensity |
| 39 | Nose Wrinkle | 0-100 | Nose wrinkling intensity |
| 40 | Smile | 0-100 | Smile intensity |
| 41 | Smirk | 0-100 | Smirk intensity |
| 42 | Upper Lip Raise | 0-100 | Upper lip raise intensity |
| 43 | Blink | 0 or 1 | Blink presence |
| 44 | BlinkRate | 0-60+ | Blinks per minute (10s window) |

### Head Rotation Columns (45-48)

| Index | Column | Unit | Description |
|-------|--------|------|-------------|
| 45 | Pitch | Degrees | Up/down rotation (+up, -down) |
| 46 | Yaw | Degrees | Left/right rotation (+right, -left) |
| 47 | Roll | Degrees | Tilt rotation (+left, -right) |
| 48 | Interocular Distance | Pixels | Distance between eyes |

### Annotation Column (49)

| Index | Column | Description |
|-------|--------|-------------|
| 49 | Respondent Annotations active | Current experiment phase annotation |

## Annotation Encoding Scheme

The annotation column uses a structured encoding to indicate the current experiment phase:

```
s{segment}r{round}m{period}{phase}
```

### Components

| Component | Description | Values |
|-----------|-------------|--------|
| `s{segment}` | Experiment segment | s1, s2, s3, s4 |
| `r{round}` | Round within segment | r1-r14 |
| `m{period}` | Period within round | m1-m15 (varies by round) |
| `{phase}` | Current activity phase | See phase table below |

> **WARNING: m1 is NOT period 1!** The `m{N}` value does NOT directly correspond to oTree period numbers. Due to pre-increment logic in the annotation generator, `m1` represents non-period phases (like SegmentIntro), and `m2` is actually oTree period 1. See the Period Offset Mapping section below for details.

### Segment Mapping

| Annotation Segment | oTree App | Has Chat |
|--------------------|-----------|----------|
| s1 | chat_noavg | No |
| s2 | chat_noavg2 | No |
| s3 | chat_noavg3 | Yes |
| s4 | chat_noavg4 | Yes |

### Phase Types

| Phase | Description | When Occurring |
|-------|-------------|----------------|
| `SegmentIntro` | Segment introduction screen | Start of each segment |
| `SegmentIntroWait` | Waiting for others | After intro |
| `MarketPeriod` | Active trading period | During market activity |
| `MarketPeriodWait` | Waiting between periods | After market period ends |
| `MarketPeriodPayoffWait` | Payoff calculation wait | After selling |
| `ResultsWait` | Waiting for results display | Before results shown |
| `Results` | Results display screen | End of round |
| `Chat` | Chat period (s3, s4 only) | Chat phase in round |
| `ChatWait` | Waiting for chat to end | After chat |
| `NewRule` | New rule introduction | When rules change |
| `NewRuleWait` | Waiting after rule intro | After NewRule |

### Period Offset Mapping

The `m{N}` value in annotations uses a **+1 offset** from oTree period numbers due to how `generate_annotations_unfiltered_v2.py` generates annotations:

1. The market period counter starts at 1
2. The counter is **pre-incremented** before recording each `MarketPeriod` event
3. Therefore, the first `MarketPeriod` (oTree period 1) receives `m2`

**Mapping Formula:**

```
iMotions m-value = oTree period + 1
oTree period = iMotions m-value - 1
```

**Mapping Table:**

| iMotions m-value | oTree Period | Phase |
|------------------|--------------|-------|
| m1 | N/A | SegmentIntro, SegmentIntroWait (before any MarketPeriod) |
| m2 | 1 | First MarketPeriod in round |
| m3 | 2 | Second MarketPeriod in round |
| m4 | 3 | Third MarketPeriod in round |
| ... | ... | ... |
| m{N} | N-1 | Nth MarketPeriod in round |

**Example:** To find facial data for oTree period 5 in round 3 of segment 2:
- Calculate m-value: `5 + 1 = 6`
- Build annotation: `s2r3m6MarketPeriod`

### Special Non-Period Annotations

| Annotation | Description |
|------------|-------------|
| `Label` | Calibration/labeling phase |
| `Allocate` | Asset allocation phase |
| `Survey` | Post-experiment survey |
| `Results` | Final results display |

## Timestamp Synchronization

### iMotions Timestamp

- **Column:** `Timestamp` (index 1)
- **Unit:** Milliseconds since recording start
- **Reference:** Unix timestamp in metadata row 9

### Converting to Absolute Time

```python
from datetime import datetime, timedelta

# From metadata row 9
recording_start = datetime(2025, 12, 5, 22, 20, 22, 911000)  # Example

# Convert iMotions timestamp to absolute time
def imotions_to_datetime(imotions_ms, recording_start):
    return recording_start + timedelta(milliseconds=imotions_ms)
```

### Matching to oTree Sell Events

To correlate iMotions data with oTree sell events:

1. **Get sell timestamp from oTree:** Available in period data
2. **Convert both to common reference:** Use Unix timestamps
3. **Find matching annotation:** Look for `MarketPeriod` phase at matching segment/round/period
4. **Extract facial data:** Filter rows within the relevant time window

## Participant ID Mapping

### File Naming Convention

Files follow the pattern: `{number}_{letter}{session_suffix}.csv`

| Component | Description |
|-----------|-------------|
| `{number}` | Participant order (001-016) |
| `{letter}` | Participant letter ID (A-R, excluding I, O) |
| `{session_suffix}` | Session number (3-8 for sessions 1-6) |

### Mapping Table

| Session | Suffix | Example File |
|---------|--------|--------------|
| 1 | 3 | 001_R3.csv |
| 2 | 4 | 001_R4.csv |
| 3 | 5 | 001_R5.csv |
| 4 | 6 | 001_R6.csv |
| 5 | 7 | 001_R7.csv |
| 6 | 8 | 001_R8.csv |

### oTree Participant Mapping

To match iMotions participants to oTree participants:

1. Extract letter ID from filename (e.g., "R" from "001_R3.csv")
2. Match to oTree `participant.label` field
3. Session number determines which oTree session data to use

## Example Data Access Pattern

```python
import pandas as pd

def load_imotions_data(session, participant_letter):
    """Load iMotions data for a specific participant."""
    suffix = session + 2  # Session 1 = suffix 3, etc.

    # Find the file with matching letter
    import glob
    pattern = f"datastore/imotions/{session}/*_{participant_letter}{suffix}.csv"
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No file found for {participant_letter} in session {session}")

    # Read CSV, skipping metadata
    df = pd.read_csv(files[0], skiprows=24, encoding='utf-8-sig')

    return df

def otree_period_to_m_value(otree_period):
    """Convert oTree period number to iMotions m-value.

    The annotation generator pre-increments the period counter,
    so m2 = oTree period 1, m3 = oTree period 2, etc.
    """
    return otree_period + 1

def filter_by_experiment_phase(df, segment, round_num, otree_period, phase):
    """Filter data to specific experiment phase.

    Args:
        df: DataFrame with iMotions data
        segment: Segment number (1-4)
        round_num: Round number within segment (1-14)
        otree_period: oTree period number (1-based, will be converted to m-value)
        phase: Phase name (e.g., 'MarketPeriod', 'Results')

    Returns:
        Filtered DataFrame for the specified phase

    Note: This function automatically applies the +1 offset to convert
    oTree periods to iMotions m-values.
    """
    m_value = otree_period_to_m_value(otree_period)
    annotation = f"s{segment}r{round_num}m{m_value}{phase}"
    return df[df['Respondent Annotations active'] == annotation]

def get_market_period_emotions(df, segment, round_num, otree_period):
    """Get emotion data during a market period.

    Args:
        df: DataFrame with iMotions data
        segment: Segment number (1-4)
        round_num: Round number within segment (1-14)
        otree_period: oTree period number (1-based)

    Returns:
        DataFrame with timestamp and emotion columns for the specified period
    """
    filtered = filter_by_experiment_phase(df, segment, round_num, otree_period, 'MarketPeriod')

    emotion_cols = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy',
                    'Sadness', 'Surprise', 'Engagement', 'Valence']

    return filtered[['Timestamp'] + emotion_cols]
```

### Direct Annotation Construction (Advanced)

If you need to build annotations manually without the helper functions:

```python
def build_annotation(segment, round_num, otree_period, phase):
    """Build annotation string from oTree coordinates.

    IMPORTANT: Always add 1 to oTree period to get the m-value.

    Examples:
        build_annotation(1, 3, 5, 'MarketPeriod')  -> 's1r3m6MarketPeriod'
        build_annotation(2, 1, 1, 'MarketPeriod')  -> 's2r1m2MarketPeriod'
    """
    m_value = otree_period + 1  # Apply the offset
    return f"s{segment}r{round_num}m{m_value}{phase}"
```

## Data Quality Considerations

### Missing Data

- **Empty annotation:** Many rows have empty annotations (pre-experiment or transitions)
- **Missing face detection:** AFFDEX may fail to detect face, resulting in NaN values
- **Blink artifacts:** Brief gaps during blinks affect some metrics

### Filtering Recommendations

1. **Attention threshold:** Filter rows where `Attention < 50` for unreliable detection
2. **Interocular distance:** Significant changes may indicate camera issues
3. **Sampling gaps:** Check for timestamp jumps > 100ms between consecutive rows

## Files Created

- `issues/issue_15_imotions_documentation.md` - This documentation file
- `.claude/skills/imotions-data/skill.md` - Claude skill for iMotions data processing
