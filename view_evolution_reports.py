#!/usr/bin/env python3
"""
Evolution Snapshot Viewer for AlpacaBot
Views the hourly JSON evolution reports showing system learning and performance
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict
import argparse

def load_evolution_snapshots(reports_dir: Path = None) -> List[Dict]:
    """Load all evolution snapshot JSON files"""
    if reports_dir is None:
        reports_dir = Path(__file__).parent / 'reports'
    
    snapshots = []
    for file in sorted(reports_dir.glob('evolution_snapshot_*.json')):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                snapshots.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return snapshots

def display_latest_snapshot(snapshots: List[Dict]):
    """Display the most recent evolution snapshot"""
    if not snapshots:
        print("No evolution snapshots found!")
        return
    
    latest = snapshots[-1]
    print("\n" + "="*80)
    print("LATEST EVOLUTION SNAPSHOT")
    print("="*80)
    print(f"Timestamp: {latest['timestamp']}")
    print(f"Runtime: {latest['runtime']}")
    print(f"File: {latest['filename']}")
    
    # Q-Learning Status
    if 'q_learning' in latest['components']:
        ql = latest['components']['q_learning']
        print("\nQ-LEARNING MODEL:")
        print(f"  Status: {ql['status']}")
        print(f"  Model Size: {ql['model_size_kb']:.2f} KB")
        print(f"  Epsilon: {ql['epsilon']:.3f}")
        print(f"  Learning Rate: {ql['learning_rate']}")
        print(f"  Last Update: {ql['last_update']}")
    
    # Predictions Performance
    if 'predictions' in latest['components']:
        pred = latest['components']['predictions']
        print("\nPREDICTIONS (Last 24h):")
        print(f"  Total Predictions: {pred['last_24h_predictions']}")
        print(f"  Avg Confidence: {pred['avg_confidence']:.1%}")
        print(f"  Accuracy Rate: {pred['accuracy_rate']:.1%}" if pred['accuracy_rate'] else "  Accuracy Rate: N/A")
        print(f"  Symbols Tracked: {pred['symbols_tracked']}")
        print(f"  Learning Updates: {pred['learning_updates']}")
    
    # Portfolio Performance
    if 'portfolio' in latest['components']:
        port = latest['components']['portfolio']
        print("\nPORTFOLIO STATUS:")
        print(f"  Current Value: ${port['current_value']:,.2f}")
        print(f"  Daily Change: ${port['daily_change']:,.2f} ({port['daily_change_pct']:.2f}%)")
        print(f"  Total Change: ${port['total_change']:,.2f} ({port['total_change_pct']:.2f}%)")
        print(f"  Active Positions: {port['positions']}")
        print(f"  Buying Power: ${port['buying_power']:,.2f}")
        print(f"  Win Rate: {port['win_rate']:.1f}%")
        print(f"  Total Trades: {port['total_trades']}")
    
    # Strategy Performance
    if 'strategies' in latest['components']:
        print("\nSTRATEGY PERFORMANCE:")
        strategies = latest['components']['strategies']
        active_strategies = []
        for name, data in strategies.items():
            if data['active']:
                active_strategies.append(name)
                if data['trades'] > 0:
                    print(f"  {name.replace('_', ' ').title()}:")
                    print(f"    Trades: {data['trades']}, Profit: ${data['profit']:,.2f}")
        
        if not any(s['trades'] > 0 for s in strategies.values()):
            print(f"  Active Strategies: {', '.join(active_strategies)}")
            print("  No trades executed yet in this session")

def display_evolution_timeline(snapshots: List[Dict]):
    """Display portfolio evolution over time"""
    if not snapshots:
        return
    
    print("\n" + "="*80)
    print("PORTFOLIO EVOLUTION TIMELINE")
    print("="*80)
    
    timeline_data = []
    for snap in snapshots:
        if 'portfolio' in snap.get('components', {}):
            port = snap['components']['portfolio']
            time_str = snap['timestamp'].split('T')[1][:8]
            timeline_data.append({
                'Time': time_str,
                'Value': f"${port['current_value']:,.0f}",
                'Daily %': f"{port['daily_change_pct']:+.2f}%",
                'Total %': f"{port['total_change_pct']:+.2f}%",
                'Trades': port['total_trades']
            })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data[-10:])  # Show last 10 snapshots
        print(df.to_string(index=False))

def display_learning_progress(snapshots: List[Dict]):
    """Display Q-learning model evolution"""
    if not snapshots:
        return
    
    print("\n" + "="*80)
    print("Q-LEARNING MODEL EVOLUTION")
    print("="*80)
    
    learning_data = []
    for snap in snapshots:
        if 'q_learning' in snap.get('components', {}):
            ql = snap['components']['q_learning']
            time_str = snap['timestamp'].split('T')[1][:8]
            learning_data.append({
                'Time': time_str,
                'Epsilon': f"{ql['epsilon']:.3f}",
                'Learning Rate': f"{ql['learning_rate']:.4f}",
                'Model Size (KB)': f"{ql['model_size_kb']:.1f}"
            })
    
    if learning_data:
        df = pd.DataFrame(learning_data[-10:])  # Show last 10 snapshots
        print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='View AlpacaBot Evolution Reports')
    parser.add_argument('--all', action='store_true', help='Show all evolution data')
    parser.add_argument('--timeline', action='store_true', help='Show portfolio timeline')
    parser.add_argument('--learning', action='store_true', help='Show Q-learning evolution')
    parser.add_argument('--latest', action='store_true', help='Show only latest snapshot (default)')
    args = parser.parse_args()
    
    # Load snapshots
    snapshots = load_evolution_snapshots()
    
    if not snapshots:
        print("No evolution snapshots found in reports directory!")
        return
    
    print(f"\nFound {len(snapshots)} evolution snapshots")
    
    # Display based on arguments
    if args.all or args.timeline:
        display_evolution_timeline(snapshots)
    
    if args.all or args.learning:
        display_learning_progress(snapshots)
    
    # Always show latest unless specific option selected
    if args.all or args.latest or (not args.timeline and not args.learning):
        display_latest_snapshot(snapshots)
    
    print("\n" + "="*80)
    print(f"System has been learning for {snapshots[-1]['runtime']}")
    print("="*80)

if __name__ == "__main__":
    main()