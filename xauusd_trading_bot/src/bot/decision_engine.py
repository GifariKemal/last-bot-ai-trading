"""
Decision Engine
Central decision-making logic that evaluates market conditions and signals.
"""

from typing import Dict, Optional, List
from datetime import datetime
from ..bot_logger import get_logger


class DecisionEngine:
    """Make trading decisions based on all available information."""

    def __init__(self, config: Dict):
        """
        Initialize decision engine.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger()
        self.config = config

        # Decision thresholds
        self.min_confidence = config.get("strategy", {}).get("entry", {}).get(
            "min_confluence_score", 0.65
        )

    def evaluate_entry_signal(
        self,
        signal: Dict,
        market_context: Dict,
        position_context: Dict,
        session_context: Dict,
        risk_context: Dict
    ) -> Dict:
        """
        Evaluate if entry signal should be executed.

        Args:
            signal: Entry signal from strategy
            market_context: Market analysis context
            position_context: Position management context
            session_context: Session management context
            risk_context: Risk management context

        Returns:
            Decision dictionary
        """
        try:
            # Check if signal is valid
            if not signal or not signal.get("valid"):
                return {
                    "execute": False,
                    "reason": signal.get("reasons", ["Invalid signal"])[0] if signal else "No signal",
                    "confidence": 0.0,
                }

            # Extract signal details
            confidence = signal.get("confidence", 0)
            direction = signal.get("direction")

            # Build decision factors
            factors = []
            score = confidence  # Start with signal confidence

            # 1. Session factor
            session_weight = session_context.get("session_weight", 1.0)
            adjusted_confidence = confidence * session_weight
            factors.append({
                "factor": "session",
                "weight": session_weight,
                "adjustment": adjusted_confidence - confidence,
                "favorable": session_context.get("is_preferred", False),
            })

            # 2. Market condition factor
            market_favorable = market_context.get("is_favorable", False)
            if not market_favorable:
                score *= 0.9  # Reduce confidence in unfavorable conditions
                factors.append({
                    "factor": "market_condition",
                    "favorable": False,
                    "adjustment": -0.1,
                })
            else:
                factors.append({
                    "factor": "market_condition",
                    "favorable": True,
                    "adjustment": 0.0,
                })

            # 3. Position management factor
            position_allowed = position_context.get("allowed", True)
            if not position_allowed:
                return {
                    "execute": False,
                    "reason": position_context.get("reason", "Position limits"),
                    "confidence": adjusted_confidence,
                    "factors": factors,
                }

            # 4. Risk factor
            risk_allowed = risk_context.get("trading_allowed", True)
            if not risk_allowed:
                return {
                    "execute": False,
                    "reason": risk_context.get("reason", "Risk limits exceeded"),
                    "confidence": adjusted_confidence,
                    "factors": factors,
                }

            # 5. Check final confidence threshold
            min_threshold = session_context.get("adjusted_confluence_threshold", self.min_confidence)

            if adjusted_confidence < min_threshold:
                return {
                    "execute": False,
                    "reason": f"Confidence {adjusted_confidence:.2f} below threshold {min_threshold:.2f}",
                    "confidence": adjusted_confidence,
                    "factors": factors,
                }

            # All checks passed - EXECUTE
            return {
                "execute": True,
                "reason": "All decision factors favorable",
                "confidence": adjusted_confidence,
                "direction": direction,
                "signal": signal,
                "factors": factors,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in decision evaluation: {e}")
            return {
                "execute": False,
                "reason": f"Decision error: {e}",
                "confidence": 0.0,
            }

    def evaluate_exit_signal(
        self,
        exit_signal: Dict,
        position: Dict,
        market_context: Dict
    ) -> Dict:
        """
        Evaluate if exit signal should be executed.

        Args:
            exit_signal: Exit signal from strategy
            position: Position data
            market_context: Market context

        Returns:
            Decision dictionary
        """
        try:
            if not exit_signal or not exit_signal.get("should_exit"):
                return {
                    "execute": False,
                    "reason": "No exit signal",
                }

            # Exit signals are generally always executed
            return {
                "execute": True,
                "reason": exit_signal.get("reason", "Exit conditions met"),
                "exit_signal": exit_signal,
                "position": position,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in exit evaluation: {e}")
            return {
                "execute": False,
                "reason": f"Exit evaluation error: {e}",
            }

    def evaluate_position_modification(
        self,
        modification: Dict,
        position: Dict,
        reason: str
    ) -> Dict:
        """
        Evaluate if position modification should be executed.

        Args:
            modification: Modification data (new SL/TP)
            position: Position data
            reason: Reason for modification

        Returns:
            Decision dictionary
        """
        try:
            if not modification or not modification.get("new_sl"):
                return {
                    "execute": False,
                    "reason": "No modification needed",
                }

            # Modifications (like trailing stops) are generally executed
            return {
                "execute": True,
                "reason": reason,
                "modification": modification,
                "position": position,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in modification evaluation: {e}")
            return {
                "execute": False,
                "reason": f"Modification evaluation error: {e}",
            }

    def prioritize_actions(
        self,
        entry_decisions: List[Dict],
        exit_decisions: List[Dict],
        modification_decisions: List[Dict]
    ) -> List[Dict]:
        """
        Prioritize and order actions to execute.

        Args:
            entry_decisions: List of entry decisions
            exit_decisions: List of exit decisions
            modification_decisions: List of modification decisions

        Returns:
            Ordered list of actions by priority
        """
        actions = []

        # Priority 1: Exit signals (close losing positions first)
        for decision in exit_decisions:
            if decision.get("execute"):
                position = decision.get("position", {})
                profit = position.get("profit", 0)
                priority = 1 if profit < 0 else 2  # Losing positions first

                actions.append({
                    "type": "exit",
                    "priority": priority,
                    "decision": decision,
                })

        # Priority 2: Position modifications (protect profits)
        for decision in modification_decisions:
            if decision.get("execute"):
                actions.append({
                    "type": "modification",
                    "priority": 3,
                    "decision": decision,
                })

        # Priority 3: Entry signals (take new opportunities)
        for decision in entry_decisions:
            if decision.get("execute"):
                actions.append({
                    "type": "entry",
                    "priority": 4,
                    "decision": decision,
                })

        # Sort by priority
        actions.sort(key=lambda x: x["priority"])

        return actions

    def get_decision_summary(self, decision: Dict) -> str:
        """
        Get human-readable decision summary.

        Args:
            decision: Decision dictionary

        Returns:
            Summary string
        """
        if not decision.get("execute"):
            return f"NO ACTION: {decision.get('reason', 'Unknown')}"

        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")

        if decision.get("direction"):
            direction = decision["direction"]
            return f"EXECUTE {direction}: {reason} (Confidence: {confidence:.2f})"

        return f"EXECUTE: {reason}"

    def should_pause_trading(
        self,
        risk_status: Dict,
        emergency_status: Dict,
        account_status: Dict
    ) -> Dict:
        """
        Check if trading should be paused.

        Args:
            risk_status: Risk management status
            emergency_status: Emergency handler status
            account_status: Account health status

        Returns:
            Pause recommendation
        """
        pause_reasons = []

        # Check emergency state
        if emergency_status.get("emergency_active"):
            pause_reasons.append("Emergency stop active")

        # Check risk limits
        if not risk_status.get("allowed"):
            pause_reasons.append(risk_status.get("reason", "Risk limits exceeded"))

        # Check account health
        if not account_status.get("healthy"):
            pause_reasons.append(account_status.get("reason", "Account unhealthy"))

        if pause_reasons:
            return {
                "should_pause": True,
                "reasons": pause_reasons,
            }

        return {
            "should_pause": False,
            "reason": "All systems normal",
        }
