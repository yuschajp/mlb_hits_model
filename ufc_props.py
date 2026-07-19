#!/usr/bin/env python3
"""
UFC Prop Betting Analysis
Detects value in common UFC props: KO/Sub props, strike totals, round overs, etc.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math

# ============================================================================
# PROP MODELS
# ============================================================================

@dataclass
class PropPrediction:
    """Single prop prediction with model probability vs. market odds"""
    prop_name: str
    fighter: str
    prediction: str  # e.g., "Over", "Under", "Yes", "No"
    model_prob: float  # Our estimated probability
    market_odds: float  # Decimal odds (1.50 = -200)
    implied_prob: float  # Market's implied probability
    value: float  # Expected value: (model_prob * odds) - 1
    edge_percent: float  # (model_prob - implied_prob) * 100
    recommendation: str  # "Strong Play", "Play", "Pass", "Fade"
    notes: str
    
    def to_dict(self):
        return asdict(self)


class UFCPropAnalyzer:
    """Analyze UFC prop bets for value"""
    
    # Minimum edge required to recommend a play (in basis points)
    MIN_EDGE_BP = 350  # 3.5% edge
    MIN_ODDS = 1.5  # Skip heavy favorites/underdogs
    MAX_ODDS = 3.5
    
    def __init__(self):
        self.fighter_stats = {}  # Cache historical fighter stats
    
    def decimal_to_american(self, decimal: float) -> int:
        """Convert decimal odds to American"""
        if decimal >= 2:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))
    
    def american_to_implied_prob(self, american: int) -> float:
        """Convert American odds to implied probability"""
        if american > 0:
            return 100 / (american + 100)
        else:
            return abs(american) / (abs(american) + 100)
    
    def decimal_to_implied_prob(self, decimal: float) -> float:
        """Convert decimal odds to implied probability"""
        return 1 / decimal
    
    def calculate_ev(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calculate expected value
        EV = (probability * odds) - 1
        Positive EV means we should take the bet
        """
        return (model_prob * decimal_odds) - 1
    
    def calculate_edge(self, model_prob: float, implied_prob: float) -> float:
        """
        Calculate our edge in basis points
        edge = (model_prob - implied_prob) * 10000
        """
        return (model_prob - implied_prob) * 10000
    
    def recommend(self, edge_bp: float, ev: float) -> str:
        """Generate recommendation based on edge and EV"""
        if edge_bp >= 600 and ev >= 0.15:
            return "Strong Play"
        elif edge_bp >= self.MIN_EDGE_BP and ev >= 0.05:
            return "Play"
        elif edge_bp <= -500:
            return "Fade"
        else:
            return "Pass"
    
    def analyze_ko_tko_prop(
        self,
        fighter: str,
        ko_tko_prob: float,
        market_odds: float,
        notes: str = ""
    ) -> PropPrediction:
        """Analyze KO/TKO prop bet"""
        implied_prob = self.decimal_to_implied_prob(market_odds)
        edge_bp = self.calculate_edge(ko_tko_prob, implied_prob)
        ev = self.calculate_ev(ko_tko_prob, market_odds)
        rec = self.recommend(edge_bp, ev)
        
        return PropPrediction(
            prop_name="KO/TKO",
            fighter=fighter,
            prediction="Yes",
            model_prob=ko_tko_prob,
            market_odds=market_odds,
            implied_prob=implied_prob,
            value=ev,
            edge_percent=edge_bp / 100,
            recommendation=rec,
            notes=notes
        )
    
    def analyze_submission_prop(
        self,
        fighter: str,
        sub_prob: float,
        market_odds: float,
        notes: str = ""
    ) -> PropPrediction:
        """Analyze submission prop bet"""
        implied_prob = self.decimal_to_implied_prob(market_odds)
        edge_bp = self.calculate_edge(sub_prob, implied_prob)
        ev = self.calculate_ev(sub_prob, market_odds)
        rec = self.recommend(edge_bp, ev)
        
        return PropPrediction(
            prop_name="Submission",
            fighter=fighter,
            prediction="Yes",
            model_prob=sub_prob,
            market_odds=market_odds,
            implied_prob=implied_prob,
            value=ev,
            edge_percent=edge_bp / 100,
            recommendation=rec,
            notes=notes
        )
    
    def analyze_round_over_under(
        self,
        fighter1: str,
        fighter2: str,
        round_total: float,
        fight_distance_rounds: int,
        model_over_prob: float,
        market_odds: float,
        prediction: str = "Over"
    ) -> PropPrediction:
        """Analyze round total Over/Under"""
        implied_prob = self.decimal_to_implied_prob(market_odds)
        edge_bp = self.calculate_edge(model_over_prob, implied_prob)
        ev = self.calculate_ev(model_over_prob, market_odds)
        rec = self.recommend(edge_bp, ev)
        
        return PropPrediction(
            prop_name=f"Rounds {round_total}",
            fighter=f"{fighter1} vs {fighter2}",
            prediction=prediction,
            model_prob=model_over_prob,
            market_odds=market_odds,
            implied_prob=implied_prob,
            value=ev,
            edge_percent=edge_bp / 100,
            recommendation=rec,
            notes=f"Fight distance: {fight_distance_rounds} rounds"
        )
    
    def analyze_significant_strikes(
        self,
        fighter: str,
        strikes_over: int,
        model_prob_over: float,
        market_odds: float
    ) -> PropPrediction:
        """Analyze significant strikes Over/Under"""
        implied_prob = self.decimal_to_implied_prob(market_odds)
        edge_bp = self.calculate_edge(model_prob_over, implied_prob)
        ev = self.calculate_ev(model_prob_over, market_odds)
        rec = self.recommend(edge_bp, ev)
        
        return PropPrediction(
            prop_name=f"Sig Strikes Over {strikes_over}",
            fighter=fighter,
            prediction="Over",
            model_prob=model_prob_over,
            market_odds=market_odds,
            implied_prob=implied_prob,
            value=ev,
            edge_percent=edge_bp / 100,
            recommendation=rec,
            notes="Based on fighter pace and opponent defense"
        )
    
    def filter_recommendations(
        self,
        props: List[PropPrediction],
        min_recommendation: str = "Play"
    ) -> List[PropPrediction]:
        """Filter props by recommendation level"""
        ranking = {"Strong Play": 3, "Play": 2, "Pass": 1, "Fade": 0}
        min_rank = ranking.get(min_recommendation, 1)
        
        return [p for p in props if ranking.get(p.recommendation, 0) >= min_rank]
    
    def generate_report(
        self,
        props: List[PropPrediction],
        event_name: str
    ) -> str:
        """Generate formatted prop analysis report"""
        lines = [
            "=" * 80,
            f"UFC PROP ANALYSIS REPORT: {event_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]
        
        # Strong plays
        strong = [p for p in props if p.recommendation == "Strong Play"]
        if strong:
            lines.append("🔥 STRONG PLAYS:")
            for p in strong:
                lines.append(f"  {p.fighter} | {p.prop_name} {p.prediction}")
                lines.append(f"    Model: {p.model_prob:.1%} | Market: {p.implied_prob:.1%}")
                lines.append(f"    Odds: {p.market_odds:.2f} | Edge: +{p.edge_percent:.1f}bp | EV: +{p.value:.2f}")
                if p.notes:
                    lines.append(f"    Note: {p.notes}")
            lines.append("")
        
        # Plays
        plays = [p for p in props if p.recommendation == "Play"]
        if plays:
            lines.append("✓ PLAYS:")
            for p in plays:
                lines.append(f"  {p.fighter} | {p.prop_name} {p.prediction}")
                lines.append(f"    Model: {p.model_prob:.1%} | Market: {p.implied_prob:.1%}")
                lines.append(f"    Odds: {p.market_odds:.2f} | Edge: +{p.edge_percent:.1f}bp")
            lines.append("")
        
        # Fades
        fades = [p for p in props if p.recommendation == "Fade"]
        if fades:
            lines.append("⚠ FADES (Bet the other side):")
            for p in fades:
                lines.append(f"  {p.fighter} | {p.prop_name} {p.prediction}")
                lines.append(f"    Model: {p.model_prob:.1%} | Market: {p.implied_prob:.1%}")
                lines.append(f"    Edge: {p.edge_percent:.1f}bp (AGAINST this prop)")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# ============================================================================
# EXAMPLE USAGE (Run standalone for testing)
# ============================================================================

def main():
    """Test prop analysis"""
    analyzer = UFCPropAnalyzer()
    
    # Example: Analyzing props for a hypothetical fight
    props = []
    
    # KO/TKO prop
    props.append(analyzer.analyze_ko_tko_prop(
        fighter="Fighter A",
        ko_tko_prob=0.42,
        market_odds=2.10,  # +110
        notes="Aggressive striker vs. defensive opponent"
    ))
    
    # Submission prop
    props.append(analyzer.analyze_submission_prop(
        fighter="Fighter B",
        sub_prob=0.18,
        market_odds=3.50,  # +250
        notes="Grappler with 60% sub rate historically"
    ))
    
    # Round total
    props.append(analyzer.analyze_round_over_under(
        fighter1="Fighter A",
        fighter2="Fighter B",
        round_total=2.5,
        fight_distance_rounds=3,
        model_over_prob=0.58,
        market_odds=1.90,
        prediction="Over"
    ))
    
    # Significant strikes
    props.append(analyzer.analyze_significant_strikes(
        fighter="Fighter A",
        strikes_over=95,
        model_prob_over=0.65,
        market_odds=1.85
    ))
    
    # Print report
    report = analyzer.generate_report(props, "Fighter A vs Fighter B")
    print(report)
    
    # Save to JSON
    with open("ufc_props_analysis.json", "w") as f:
        json.dump([p.to_dict() for p in props], f, indent=2)
    
    print(f"\n✓ Analysis saved to ufc_props_analysis.json")


if __name__ == "__main__":
    main()
