# MarketRuns

A real-time market simulation game that explores the dynamics of information cascades and market behavior. Players participate in a market where they must make decisions based on their beliefs about the market state and the actions of other participants.

## Overview

MarketRuns is an interactive web-based game that simulates a market environment where:
- Players receive private signals about the market state
- A real-time market price updates based on trading activity
- Players can observe others' actions and update their beliefs
- A chat system allows for communication between participants
- Visual charts display price movements and belief updates

## Features

- **Real-time Market Simulation**: Dynamic price updates based on market activity
- **Interactive UI**: 
  - Live price and belief charts
  - Player status tracking
  - Real-time chat system
  - Visual feedback for market events
- **Belief Updating**: Players can update their beliefs based on market signals and other players' actions
- **Trading Interface**: Simple one-click trading mechanism with a 10-second initial waiting period
- **Visual Analytics**: 
  - Price movement charts
  - Belief updating charts
  - Player status indicators

## Technical Stack

- Python backend
- HTML/CSS/JavaScript frontend
- Chart.js for data visualization
- Real-time updates using WebSocket technology

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/calebeynon/MarketRuns.git
   cd MarketRuns
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.lock
   ```

4. Run the development server:
   ```bash
   python src/manage.py runserver
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8000`
2. Join a game session
3. Observe the market price and other players' actions
4. Use the chat system to communicate with other participants
5. Make trading decisions based on your beliefs and market observations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License


