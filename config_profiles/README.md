# Configuration Profiles

This directory contains pre-configured trading profiles for different risk tolerances and strategies.

## Available Profiles

### 🛡️ Conservative (`conservative.json`)
- Low risk, stable returns
- Focus on mean reversion and statistical arbitrage
- Smaller position sizes and tighter stops

### ⚡ Aggressive (`aggressive.json`)
- High risk, high reward
- Focus on momentum and news-based strategies
- Larger positions and wider stops

### ⚖️ Balanced (`balanced.json`)
- Balanced approach with all strategies
- Medium position sizes
- Default recommended profile

## Usage

To load a profile, use the following in your Python code:

```python
from config import load_profile

# Load the conservative profile
load_profile('conservative')
```

Or select it from the main menu when running the system.

## Creating Custom Profiles

You can create your own profiles by copying one of the existing profiles and modifying the parameters to suit your needs.
