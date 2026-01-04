# Sheep vs Wolf Population Simulation

A mini-web application built with SimPy and Python that simulates the evolution of sheep and wolf populations over time using discrete-event simulation.

## Features

- **Interactive Simulation**: Adjust starting populations and various parameters to see how they affect population dynamics
- **Real-time Visualization**: Beautiful charts showing population evolution over time using Chart.js
- **Configurable Parameters**:
  - Initial sheep and wolf populations
  - Birth rates for both species
  - Natural death rates
  - Predation rate (how many sheep wolves kill)
  - Carrying capacity (maximum sustainable sheep population)
  - Simulation duration

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## How It Works

The simulation uses SimPy's discrete-event simulation framework to model:

- **Sheep Population**: 
  - Reproduces based on birth rate and carrying capacity
  - Dies from natural causes and predation by wolves
  
- **Wolf Population**:
  - Reproduces based on birth rate and food availability (sheep population)
  - Dies from natural causes and starvation when food is scarce

The model demonstrates classic predator-prey dynamics, showing how populations oscillate and interact over time.

## Parameters Explained

- **Birth Rates**: Probability of reproduction per time unit
- **Death Rates**: Probability of natural death per time unit
- **Predation Rate**: How effectively wolves hunt sheep
- **Carrying Capacity**: Maximum number of sheep the environment can sustain
- **Duration**: How long to run the simulation (in time units)

## Example Scenarios

- **High Sheep Birth Rate**: Sheep population grows quickly, supporting more wolves
- **High Predation Rate**: Wolves kill more sheep, potentially leading to population crashes
- **Low Carrying Capacity**: Limits sheep growth, affecting wolf population indirectly
- **Balanced Parameters**: Creates oscillating populations typical of predator-prey systems

