# MarketRuns

An experimental economics platform built with oTree for studying market dynamics, information cascades, and trading behavior in controlled laboratory settings. The platform focuses on multi-period market experiments with integrated chat systems and real-time signal processing.

## Overview

MarketRuns is a comprehensive experimental platform designed for behavioral economics research. The **`nonlivegame`** folder contains the main experimental framework that enables researchers to:

- Conduct controlled market experiments with 4-16 participants
- Study information cascades and herding behavior in trading environments  
- Implement various chat and no-chat treatment conditions
- Collect detailed behavioral and decision-making data
- Run multi-period experiments with dynamic signal updating

## Experimental Design (nonlivegame)

The `nonlivegame` directory contains multiple experimental treatments:

### Core Game Mechanics
- **4 players per group** organized into market sessions
- **Variable number of periods** (geometric distribution, mean 8 periods, max 14)
- **Binary state environment** with probabilistic signals (67.5% accuracy)
- **Dynamic pricing** based on market activity and selling behavior
- **Bayesian signal updating** for belief formation

### Treatment Variations

1. **`chat_noavg`** - Primary treatment with chat enabled
2. **`chat_noavg2`** - Secondary chat treatment variant  
3. **`chat_noavg3`** - Third chat treatment variant
4. **`chat_noavg4`** - Fourth chat treatment variant
5. **`game`** - Base game without specific treatment conditions
6. **`quiz`** - Pre-experiment comprehension quiz
7. **`final`** - Post-experiment final results and payoffs

### Experimental Flow

```
Quiz → Chat Treatment 1 → Chat Treatment 2 → Chat Treatment 3 → Chat Treatment 4 → Final Results
```

## Key Features

### Market Mechanics
- **Signal-based Trading**: Players receive private signals about market state
- **Price Discovery**: Market prices adjust based on selling activity (-$2 per seller)
- **Payoff Structure**: 
  - Sellers receive decreasing prices based on order (first seller gets highest price)
  - Non-sellers receive liquidation value based on true market state
- **Real-time Visualization**: Dynamic price and signal history charts

### Experimental Controls
- **Participant Grouping**: Fixed grouping across treatments for within-subject analysis
- **Random State Assignment**: Binary market state (0 or 1) determined randomly
- **Standardized Interface**: Professional UI with consistent styling across treatments
- **Data Collection**: Comprehensive logging of decisions, timing, and chat interactions

### Interactive Elements
- **Chat System**: Real-time communication between participants (45-second initial chat period)
- **Visual Analytics**: Live Chart.js visualizations of price and signal histories
- **Decision Interface**: One-click selling mechanism with 10-second decision windows
- **Results Feedback**: Period-by-period results showing market outcomes

## Technical Architecture

### Backend (oTree Framework)
- **Python-based**: Built on the oTree experimental economics platform
- **Session Management**: Handles participant flow, grouping, and data persistence
- **Bayesian Updating**: Implements optimal signal processing algorithms
- **Payment System**: Integrated payoff calculation and participant compensation

### Frontend Technologies
- **HTML/CSS**: Professional, responsive interface design
- **JavaScript**: Real-time chart updates and interactive elements
- **Chart.js**: Dynamic visualization of market data
- **Bootstrap-style**: Modern, accessible UI components

## Setup and Installation

### Prerequisites
- Python 3.8+
- Rye package manager (recommended)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/calebeynon/MarketRuns.git
   cd MarketRuns
   ```

2. **Install dependencies using Rye:**
   ```bash
   rye sync
   ```

3. **Navigate to experimental directory:**
   ```bash
   cd nonlivegame
   ```

4. **Run the oTree server:**
   ```bash
   otree devserver
   ```

5. **Access the experiment:**
   - Open browser to `http://localhost:8000`
   - Use the admin interface to create and manage sessions

### Alternative Installation (pip)

If not using Rye:
```bash
pip install -r requirements.lock
cd nonlivegame  
otree devserver
```

## Running Experiments

### Session Configuration
- **Participants**: 16 participants (4 groups of 4)
- **Duration**: ~45-60 minutes depending on number of periods
- **Payments**: $7.50 participation fee + variable performance payments
- **Room Setup**: Uses `participant_labels.txt` for participant identification

### Admin Interface
1. Navigate to the admin panel
2. Create a new session using the `chat_noavg` configuration
3. Generate participant links
4. Monitor session progress in real-time
5. Export data after completion

## Data Collection

The platform automatically collects:
- **Decision Data**: Selling decisions, timing, and payoffs
- **Signal History**: Complete record of private signals and belief updates
- **Price Dynamics**: Market price evolution across periods
- **Chat Logs**: Full transcripts of participant communications
- **Demographics**: Basic participant information and quiz responses

## Research Applications

This platform is suitable for studying:
- **Information Cascades**: How private information propagates through markets
- **Herding Behavior**: Social influence on trading decisions
- **Communication Effects**: Impact of chat on market efficiency
- **Signal Processing**: Individual vs. social learning in uncertain environments
- **Market Microstructure**: Price formation in thin markets

## Contributing

Contributions welcome! Areas for development:
- Additional treatment variations
- Enhanced data visualization
- Mobile-responsive improvements
- Advanced statistical analysis tools

Please submit Pull Requests with detailed descriptions of changes.

## License

MIT License - See LICENSE file for details.

