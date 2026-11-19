"""
fuzzy_system.py
===============
Mamdani-style Fuzzy Inference System for steel surface inspection decision support.

Inputs  : confidence_prob (float 0-1), normalized_area (float 0-1), predicted_class (str)
Output  : decision (Accept/Monitor/Rework/Reject), risk_score (float 0-1)

Membership functions:
  trapmf(x, a, b, c, d)  - trapezoidal
  trimf(x, a, b, c)      - triangular

Risk score defuzzification:
  risk = 0.40 * confidence + 0.30 * norm_area + 0.30 * class_severity
  with rule-based threshold adjustments
"""

from __future__ import annotations

from typing import Dict, Tuple


class SteelInspectionFuzzySystem:
    """Mamdani-style fuzzy inference: (confidence, area, class) -> (decision, risk_score).

    CLASS_SEVERITY_WEIGHTS: structural risk weight per defect class [0,1].
    Higher weight = more critical defect type.
    """

    CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
        "Stains":       0.35,
        "Dirtiness":    0.40,
        "Other_Faults": 0.55,
        "Bumps":        0.70,
        "Pastry":       0.75,
        "Z_Scratch":    0.88,
        "K_Scatch":     0.95,
    }

    @staticmethod
    def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function.

        trapmf(x; a,b,c,d):
          = 0            if x <= a or x >= d
          = (x-a)/(b-a)  if a < x < b    [rising ramp]
          = 1            if b <= x <= c   [plateau]
          = (d-x)/(d-c)  if c < x < d    [falling ramp]
        """
        eps = 1e-9
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return float((x - a) / max(b - a, eps))
        return float((d - x) / max(d - c, eps))

    @staticmethod
    def trimf(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function.

        trimf(x; a,b,c):
          = 0            if x <= a or x >= c
          = (x-a)/(b-a)  if a < x <= b   [rising side]
          = (c-x)/(c-b)  if b < x < c    [falling side]
        """
        eps = 1e-9
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return float((x - a) / max(b - a, eps))
        return float((c - x) / max(c - b, eps))

    def evaluate_decision(
        self,
        predicted_class_name: str,
        confidence_prob: float,
        normalized_area: float,
    ) -> Tuple[str, float]:
        """Evaluate fuzzy rule base and return (decision, risk_score).

        Parameters
        ----------
        predicted_class_name : one of 7 defect class strings
        confidence_prob      : model softmax confidence [0, 1]
        normalized_area      : (LogOfAreas - global_min) / (global_max - global_min)

        Returns
        -------
        decision   : 'Accept' | 'Monitor' | 'Rework' | 'Reject'
        risk_score : float in [0, 1]
        """
        # Step 1: Fuzzify confidence
        prob_low  = self.trapmf(confidence_prob, 0.0, 0.0, 0.35, 0.55)
        prob_med  = self.trimf( confidence_prob, 0.40, 0.65, 0.85)
        prob_high = self.trapmf(confidence_prob, 0.70, 0.85, 1.0, 1.0)

        # Step 2: Fuzzify area severity
        area_small = self.trapmf(normalized_area, 0.0, 0.0, 0.25, 0.45)
        area_mod   = self.trimf( normalized_area, 0.30, 0.55, 0.75)
        area_large = self.trapmf(normalized_area, 0.65, 0.80, 1.0, 1.0)

        # Step 3: Get class structural risk weight
        class_risk = self.CLASS_SEVERITY_WEIGHTS.get(predicted_class_name, 0.55)

        # Step 4: Rule firing strengths (Mamdani min-operator)
        r_accept   = min(prob_low,  area_small) * (1.0 - 0.5 * class_risk)
        _r_monitor = max(min(prob_med, area_small), min(prob_low,  area_mod))
        _r_rework  = max(min(prob_high, area_mod), min(prob_med,  area_large)) * class_risk
        r_reject   = min(prob_high, area_large) * (0.6 + 0.4 * class_risk)

        # Step 5: Weighted risk score (defuzzification)
        risk_score = (
            0.40 * confidence_prob
            + 0.30 * normalized_area
            + 0.30 * class_risk
        )
        # Rule-based threshold adjustments
        if r_reject > 0.5:
            risk_score = max(risk_score, 0.82)
        elif r_accept > 0.5 and class_risk <= 0.45:
            risk_score = min(risk_score, 0.25)

        # Step 6: Crisp decision from risk score
        if risk_score < 0.30:
            decision = "Accept"
        elif risk_score < 0.55:
            decision = "Monitor"
        elif risk_score < 0.78:
            decision = "Rework"
        else:
            decision = "Reject"

        return decision, round(float(risk_score), 4)
