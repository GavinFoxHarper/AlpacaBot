# LAEF ML Models Directory

This directory stores trained machine learning models for the LAEF trading system.

## File Structure

- laef_agent.keras: Main Q-learning agent model
- pattern_recognition.keras: Market pattern detection model
- price_prediction.keras: Price prediction model

Models are automatically saved and loaded by the trading system. Each model includes:
- Network architecture
- Trained weights
- Training history
- Optimization parameters

## Model Management

Models are updated through:
1. Live market learning
2. Backtest optimization
3. Manual training sessions

Use LAEF's model management tools to:
- View model details
- Check training status
- Evaluate performance
- Compare versions