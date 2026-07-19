#!/usr/bin/env python3
"""
UFC Dashboard Aggregator
Pulls UFC predictions and props into unified format for dashboard display
Follows same structure as MLB, WC, F1 dashboards
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class UFCDashboard:
    """Aggregate UFC picks for dashboard"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "sport": "UFC",
            "predictions": [],
            "props": [],
            "summary": {}
        }
    
    def load_predictions(self, filepath: Optional[str] = None) -> List[Dict]:
        """Load UFC predictions from JSON"""
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Otherwise, find latest ufc_predictions file
        import glob
        files = glob.glob(os.path.join(self.data_dir, "ufc_predictions_*.json"))
        if files:
            latest = sorted(files)[-1]
            with open(latest, 'r') as f:
                return json.load(f)
        return []
    
    def load_props(self, filepath: Optional[str] = None) -> List[Dict]:
        """Load UFC props from JSON"""
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        if os.path.exists("ufc_props_analysis.json"):
            with open("ufc_props_analysis.json", 'r') as f:
                return json.load(f)
        return []
    
    def build_predictions_table(self, predictions: List[Dict]) -> pd.DataFrame:
        """Build DataFrame for predictions"""
        if not predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions)
        df = df[[
            'event', 'fighter1', 'fighter1_win_prob', 'fighter2', 'fighter2_win_prob',
            'favorite', 'prediction_confidence'
        ]]
        df.columns = [
            'Event', 'Fighter 1', 'P1 Win %', 'Fighter 2', 'P2 Win %',
            'Predicted Winner', 'Confidence'
        ]
        
        # Format percentages
        for col in ['P1 Win %', 'P2 Win %', 'Confidence']:
            df[col] = df[col].apply(lambda x: f"{x:.1%}")
        
        return df
    
    def build_props_table(self, props: List[Dict]) -> pd.DataFrame:
        """Build DataFrame for props"""
        if not props:
            return pd.DataFrame()
        
        df = pd.DataFrame(props)
        
        # Filter to only recommended plays
        df = df[df['recommendation'].isin(['Strong Play', 'Play'])]
        
        if df.empty:
            return df
        
        df = df[[
            'prop_name', 'fighter', 'prediction', 'model_prob', 'implied_prob',
            'market_odds', 'edge_percent', 'recommendation', 'value'
        ]]
        df.columns = [
            'Prop', 'Fighter', 'Pick', 'Model %', 'Market %',
            'Odds', 'Edge bp', 'Rec', 'Value'
        ]
        
        # Format columns
        for col in ['Model %', 'Market %']:
            df[col] = df[col].apply(lambda x: f"{x:.1%}")
        df['Odds'] = df['Odds'].apply(lambda x: f"{x:.2f}")
        df['Edge bp'] = df['Edge bp'].apply(lambda x: f"{x:.0f}")
        df['Value'] = df['Value'].apply(lambda x: f"{x:+.2f}")
        
        return df
    
    def generate_html_report(self, predictions: List[Dict], props: List[Dict]) -> str:
        """Generate HTML dashboard for UFC picks"""
        html_parts = [
            """
<!DOCTYPE html>
<html>
<head>
    <title>Quantified Edge - UFC Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; margin: 20px; background: #0a0e27; color: #e0e0e0; }
        h1 { color: #ffd700; border-bottom: 2px solid #ffd700; padding-bottom: 10px; }
        h2 { color: #ffd700; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #1a1f3a; }
        th { background: #1B3A5C; color: #C9A84C; padding: 12px; text-align: left; font-weight: 600; }
        td { padding: 10px 12px; border-bottom: 1px solid #2a2f4a; }
        tr:hover { background: #222639; }
        .timestamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .section { margin: 30px 0; }
        .strong-play { color: #00ff00; font-weight: bold; }
        .play { color: #ffff00; }
        .fade { color: #ff6b6b; }
        .pass { color: #888; }
        .summary { background: #1a1f3a; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🥊 Quantified Edge - UFC Dashboard</h1>
    <div class="timestamp">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
"""
        ]
        
        # Predictions section
        pred_df = self.build_predictions_table(predictions)
        if not pred_df.empty:
            html_parts.append("<h2>📊 Fight Predictions</h2>")
            html_parts.append(f"<p>Total fights: {len(predictions)}</p>")
            html_parts.append(pred_df.to_html(index=False, escape=False))
        
        # Props section
        props_df = self.build_props_table(props)
        if not props_df.empty:
            html_parts.append("<h2>💰 Prop Plays</h2>")
            strong_plays = len(props_df[props_df['Rec'] == 'Strong Play'])
            plays = len(props_df[props_df['Rec'] == 'Play'])
            html_parts.append(f"<p>Strong Plays: {strong_plays} | Plays: {plays}</p>")
            html_parts.append(props_df.to_html(index=False, escape=False))
        else:
            html_parts.append("<h2>💰 Prop Plays</h2>")
            html_parts.append("<p>No plays generated (insufficient edge).</p>")
        
        # Footer
        html_parts.append("""
    <div class="summary">
        <strong>Notes:</strong>
        <ul>
            <li>Predictions based on ELO rating system</li>
            <li>Props require minimum 3.5% edge for recommendation</li>
            <li>All odds in decimal format</li>
            <li>Update logs: ~/Desktop/mlb_hits_model/logs/</li>
        </ul>
    </div>
</body>
</html>
""")
        
        return "\n".join(html_parts)
    
    def build_dashboard(self):
        """Build and save complete dashboard"""
        predictions = self.load_predictions()
        props = self.load_props()
        
        # Generate HTML
        html = self.generate_html_report(predictions, props)
        
        output_file = "ufc_dashboard.html"
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✓ Dashboard saved to {output_file}")
        return output_file
    
    def print_summary(self):
        """Print summary to terminal"""
        predictions = self.load_predictions()
        props = self.load_props()
        
        print("\n" + "=" * 70)
        print("UFC DASHBOARD SUMMARY")
        print("=" * 70)
        
        if predictions:
            print(f"\n📊 Predictions: {len(predictions)} fights")
            for pred in predictions[:3]:
                print(f"  {pred['fighter1']} ({pred['fighter1_win_prob']:.1%}) vs "
                      f"{pred['fighter2']} ({pred['fighter2_win_prob']:.1%})")
        
        if props:
            strong = [p for p in props if p.get('recommendation') == 'Strong Play']
            plays = [p for p in props if p.get('recommendation') == 'Play']
            print(f"\n💰 Props: {len(strong)} strong plays, {len(plays)} plays")
            for prop in strong[:2]:
                print(f"  {prop['fighter']} - {prop['prop_name']} {prop['prediction']} "
                      f"(+{prop['edge_percent']:.0f}bp)")
        
        print("\n" + "=" * 70)


def main():
    dashboard = UFCDashboard()
    dashboard.build_dashboard()
    dashboard.print_summary()


if __name__ == "__main__":
    main()
