#!/usr/bin/env python3
"""
UFC Fight Predictor using ELO + Win Probability Model
Integrated with Odds API for live prop data
"""

import os
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import pandas as pd

# ============================================================================
# ELO SYSTEM
# ============================================================================

class UFCEloRating:
    """Fighter ELO rating system for UFC"""
    
    K_FACTOR = 32  # Standard K-factor; adjust for divisional depth
    BASE_RATING = 1600
    
    def __init__(self, fighter_name: str, rating: float = BASE_RATING, wins: int = 0, losses: int = 0):
        self.fighter_name = fighter_name
        self.rating = rating
        self.wins = wins
        self.losses = losses
        self.last_updated = datetime.now()
    
    def expected_score(self, opponent_rating: float) -> float:
        """Calculate expected win probability"""
        return 1 / (1 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update_rating(self, opponent_rating: float, result: int, ko_tko: bool = False, submission: bool = False):
        """
        Update rating after fight
        result: 1 = win, 0 = loss, 0.5 = draw
        ko_tko/submission: bonus multiplier for decisive wins
        """
        expected = self.expected_score(opponent_rating)
        k = self.K_FACTOR
        
        # Bonus multiplier for dominant wins
        if result == 1:
            if ko_tko or submission:
                k *= 1.5
            self.wins += 1
        elif result == 0:
            self.losses += 1
        
        rating_delta = k * (result - expected)
        self.rating += rating_delta
        self.rating = max(800, self.rating)  # Floor at 800
        self.last_updated = datetime.now()
        
        return rating_delta
    
    def to_dict(self) -> dict:
        return {
            "fighter": self.fighter_name,
            "rating": round(self.rating, 1),
            "wins": self.wins,
            "losses": self.losses,
            "record": f"{self.wins}-{self.losses}",
            "last_updated": self.last_updated.isoformat()
        }


# ============================================================================
# UFC DATA HANDLER
# ============================================================================

class UFCDataHandler:
    """Fetch and cache UFC event/fight data"""
    
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "mma_mixed_martial_arts"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.cache_dir = "ufc_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_upcoming_events(self) -> List[Dict]:
        """Fetch upcoming UFC events from Odds API"""
        try:
            url = f"{self.ODDS_API_BASE}/sports/{self.SPORT_KEY}/events"
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "h2h,moneyline"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            events = response.json()
            return events
        except Exception as e:
            print(f"Error fetching UFC events: {e}")
            return []
    
    def get_fight_odds(self, event_id: str) -> Dict:
        """Fetch odds for a specific event"""
        try:
            url = f"{self.ODDS_API_BASE}/sports/{self.SPORT_KEY}/events/{event_id}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "h2h,moneyline,spreads,totals"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching odds for event {event_id}: {e}")
            return {}
    
    def load_elo_ratings(self, filepath: str = "ufc_elo_ratings.json") -> Dict[str, UFCEloRating]:
        """Load fighter ELO ratings from JSON"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    ratings = {}
                    for fighter_name, fighter_data in data.items():
                        elo = UFCEloRating(
                            fighter_name,
                            rating=fighter_data.get("rating", UFCEloRating.BASE_RATING),
                            wins=fighter_data.get("wins", 0),
                            losses=fighter_data.get("losses", 0)
                        )
                        ratings[fighter_name] = elo
                    return ratings
            except Exception as e:
                print(f"Error loading ELO ratings: {e}")
        return {}
    
    def save_elo_ratings(self, ratings: Dict[str, UFCEloRating], filepath: str = "ufc_elo_ratings.json"):
        """Save fighter ELO ratings to JSON"""
        try:
            data = {name: elo.to_dict() for name, elo in ratings.items()}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving ELO ratings: {e}")


# ============================================================================
# FIGHT PREDICTION ENGINE
# ============================================================================

class UFCPredictor:
    """Main UFC fight prediction engine"""
    
    def __init__(self, elo_ratings: Dict[str, UFCEloRating]):
        self.elo_ratings = elo_ratings
    
    def get_or_create_fighter(self, fighter_name: str) -> UFCEloRating:
        """Get existing fighter or create new one with base rating"""
        if fighter_name not in self.elo_ratings:
            self.elo_ratings[fighter_name] = UFCEloRating(fighter_name)
        return self.elo_ratings[fighter_name]
    
    def predict_fight(self, fighter1: str, fighter2: str) -> Dict:
        """
        Predict fight outcome based on ELO ratings
        Returns win probabilities for both fighters
        """
        f1_elo = self.get_or_create_fighter(fighter1)
        f2_elo = self.get_or_create_fighter(fighter2)
        
        f1_win_prob = f1_elo.expected_score(f2_elo.rating)
        f2_win_prob = 1 - f1_win_prob
        
        return {
            "fighter1": fighter1,
            "fighter1_rating": round(f1_elo.rating, 1),
            "fighter1_win_prob": round(f1_win_prob, 4),
            "fighter2": fighter2,
            "fighter2_rating": round(f2_elo.rating, 1),
            "fighter2_win_prob": round(f2_win_prob, 4),
            "favorite": fighter1 if f1_win_prob > 0.5 else fighter2,
            "prediction_confidence": round(max(f1_win_prob, f2_win_prob), 4)
        }
    
    def predict_method_of_victory(self, fighter: str, opponent: str) -> Dict:
        """
        Rough estimate of method probabilities based on fighter style
        In production, would use historical fight data + strike/submission analytics
        """
        fighter_elo = self.get_or_create_fighter(fighter)
        
        # Placeholder: could incorporate historical KO%, sub%, decision% from fight stats
        # For now: higher rated fighters slightly favored for all outcomes
        rating_bonus = (fighter_elo.rating - 1600) / 1000 * 0.05  # Adjust probabilities slightly
        
        return {
            "fighter": fighter,
            "ko_tko_prob": round(0.35 + rating_bonus, 4),
            "submission_prob": round(0.25 + rating_bonus, 4),
            "decision_prob": round(0.40 - rating_bonus * 2, 4)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Daily UFC prediction run"""
    print("=" * 70)
    print("UFC DAILY PREDICTION MODEL")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize
    handler = UFCDataHandler()
    elo_ratings = handler.load_elo_ratings()
    predictor = UFCPredictor(elo_ratings)
    
    # Fetch upcoming events
    print("\n[1/3] Fetching upcoming UFC events...")
    events = handler.get_upcoming_events()
    
    if not events:
        print("No upcoming events found or API error.")
        return
    
    print(f"Found {len(events)} upcoming event(s)")
    
    # Build predictions
    print("\n[2/3] Generating predictions...")
    predictions = []
    
    for event in events[:5]:  # Top 5 events
        event_id = event.get("id")
        event_name = event.get("away_team", "TBA") + " vs " + event.get("home_team", "TBA")
        event_date = event.get("commence_time", "TBA")
        
        print(f"\nEvent: {event_name}")
        print(f"Date: {event_date}")
        
        # Predict main fighters
        try:
            fighter1 = event.get("away_team", "Fighter A")
            fighter2 = event.get("home_team", "Fighter B")
            prediction = predictor.predict_fight(fighter1, fighter2)
            prediction["event"] = event_name
            prediction["event_date"] = event_date
            predictions.append(prediction)
            
            print(f"  {prediction['fighter1']} ({prediction['fighter1_win_prob']:.1%}) vs "
                  f"{prediction['fighter2']} ({prediction['fighter2_win_prob']:.1%})")
            print(f"  → Predicted: {prediction['favorite']}")
            print(f"  → Confidence: {prediction['prediction_confidence']:.1%}")
        except Exception as e:
            print(f"  Error predicting: {e}")
    
    # Save results
    print("\n[3/3] Saving predictions...")
    output_file = f"ufc_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    handler.save_elo_ratings(elo_ratings)
    
    print(f"\n✓ Predictions saved to {output_file}")
    print(f"✓ ELO ratings updated ({len(elo_ratings)} fighters tracked)")
    print("=" * 70)
    
    return predictions


if __name__ == "__main__":
    main()
