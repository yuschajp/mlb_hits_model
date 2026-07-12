"""
prop_models.py

Extends tennis predictions with prop bet models:
- Game totals (over/under)
- Set spreads (e.g., "2-0 or 0-2" probability)
- Match goes to N sets

Uses Elo ratings + Monte Carlo simulation.
"""

import math
import random
from collections import defaultdict
from typing import Tuple, Dict, List

INITIAL_ELO = 1500.0
BASE_GAME_WIN_PROB = 0.60
EXPECTED_GAMES_PER_SET = 10.5
STD_GAMES_PER_SET = 2.0


def game_win_probability(elo_a: float, elo_b: float) -> float:
    """Probability that player A wins a single game, given Elo ratings."""
    expected = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    return max(0.01, min(0.99, expected))


def simulate_set(p_a_game_win: float, num_sims: int = 1000) -> Dict[Tuple[int, int], float]:
    """Monte Carlo simulation of a single set outcome."""
    outcomes = defaultdict(int)
    
    for _ in range(num_sims):
        games_a, games_b = 0, 0
        
        while True:
            if random.random() < p_a_game_win:
                games_a += 1
            else:
                games_b += 1
            
            if games_a >= 6 and games_a - games_b >= 2:
                outcomes[(games_a, games_b)] += 1
                break
            elif games_b >= 6 and games_b - games_a >= 2:
                outcomes[(games_a, games_b)] += 1
                break
            elif games_a == 6 and games_b == 6:
                if random.random() < 0.5:
                    outcomes[(7, 6)] += 1
                else:
                    outcomes[(6, 7)] += 1
                break
    
    return {k: v / num_sims for k, v in outcomes.items()}


def simulate_match(p_a_game_win: float, best_of: int = 3, num_sims: int = 1000) -> Dict[Tuple[int, int], float]:
    """Monte Carlo simulation of full match."""
    sets_to_win = (best_of // 2) + 1
    outcomes = defaultdict(int)
    
    for _ in range(num_sims):
        sets_a, sets_b = 0, 0
        
        while sets_a < sets_to_win and sets_b < sets_to_win:
            set_outcome = simulate_set(p_a_game_win, num_sims=100)
            r = random.random()
            cum_prob = 0
            for (g_a, g_b), prob in set_outcome.items():
                cum_prob += prob
                if r < cum_prob:
                    if g_a > g_b:
                        sets_a += 1
                    else:
                        sets_b += 1
                    break
        
        outcomes[(sets_a, sets_b)] += 1
    
    return {k: v / num_sims for k, v in outcomes.items()}


def prop_game_total_ou(p_a_game_win: float, ou_line: float = 38.5, num_sims: int = 1000) -> Dict[str, float]:
    """Probability of match going over/under a given game total."""
    total_games_list = []
    
    for _ in range(num_sims):
        games_a, games_b = 0, 0
        sets_a, sets_b = 0, 0
        sets_to_win = 2
        
        while sets_a < sets_to_win and sets_b < sets_to_win:
            set_outcome = simulate_set(p_a_game_win, num_sims=50)
            r = random.random()
            cum_prob = 0
            for (g_a, g_b), prob in set_outcome.items():
                cum_prob += prob
                if r < cum_prob:
                    games_a += g_a
                    games_b += g_b
                    if g_a > g_b:
                        sets_a += 1
                    else:
                        sets_b += 1
                    break
        
        total_games_list.append(games_a + games_b)
    
    over_count = sum(1 for tg in total_games_list if tg > ou_line)
    under_count = len(total_games_list) - over_count
    avg_total = sum(total_games_list) / len(total_games_list)
    
    return {
        "over": round(over_count / len(total_games_list), 4),
        "under": round(under_count / len(total_games_list), 4),
        "expected_total": round(avg_total, 1),
        "ou_line": ou_line,
    }


def prop_set_score(p_a_game_win: float, best_of: int = 3, num_sims: int = 1000) -> Dict[str, float]:
    """Probability of each possible set score outcome."""
    sets_to_win = (best_of // 2) + 1
    match_outcomes = simulate_match(p_a_game_win, best_of=best_of, num_sims=num_sims)
    
    result = {}
    total_a_wins = 0
    total_b_wins = 0
    
    for (sets_a, sets_b), prob in match_outcomes.items():
        if sets_a > sets_b:
            key = f"p_a_{sets_a}_{sets_b}"
            total_a_wins += prob
        else:
            key = f"p_b_{sets_b}_{sets_a}"
            total_b_wins += prob
        result[key] = round(prob, 4)
    
    result["match_outcome"] = {
        "a_wins": round(total_a_wins, 4),
        "b_wins": round(total_b_wins, 4),
    }
    return result


def prop_goes_to_n_sets(p_a_game_win: float, best_of: int = 3, num_sims: int = 1000) -> Dict[str, float]:
    """Probability match ends in each possible number of sets."""
    sets_to_win = (best_of // 2) + 1
    match_outcomes = simulate_match(p_a_game_win, best_of=best_of, num_sims=num_sims)
    
    sets_played = defaultdict(float)
    for (sets_a, sets_b), prob in match_outcomes.items():
        total_sets = sets_a + sets_b
        sets_played[total_sets] += prob
    
    return {f"sets_{n}": round(prob, 4) for n, prob in sorted(sets_played.items())}


def predict_props(player_a: str, player_b: str, elo_a: float, elo_b: float,
                  best_of: int = 3, num_sims: int = 1000) -> Dict:
    """Generates all prop predictions for a match."""
    p_a_game_win = game_win_probability(elo_a, elo_b)
    p_b_game_win = 1.0 - p_a_game_win
    
    return {
        "player_a": player_a,
        "player_b": player_b,
        "elo_a": round(elo_a, 1),
        "elo_b": round(elo_b, 1),
        "game_win_prob_a": round(p_a_game_win, 4),
        "game_win_prob_b": round(p_b_game_win, 4),
        "game_total_ou": prop_game_total_ou(p_a_game_win, ou_line=38.5, num_sims=num_sims),
        "set_scores": prop_set_score(p_a_game_win, best_of=best_of, num_sims=num_sims),
        "sets_played": prop_goes_to_n_sets(p_a_game_win, best_of=best_of, num_sims=num_sims),
    }
