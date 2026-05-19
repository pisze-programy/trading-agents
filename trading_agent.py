import json
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from payload_builder import MarketPayload

load_dotenv()
client = genai.Client()

class AgentDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Execution action. Choose BUY for new long, SELL for new short, HOLD for no action"
    )
    direction: Literal["LONG", "SHORT", "NONE"] = Field(
        description="Position direction. Must be NONE if action is HOLD"
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop loss price level. Must be null if action is HOLD"
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Primary target level. Must be null if action is HOLD"
    )
    position_size_pct: int = Field(
        ge=0, le=100,
        description="Position size. Must be exactly 0 if action is HOLD. Otherwise 50 or 100"
    )
    confidence: int = Field(
        ge=1, le=10,
        description="Conviction score 1-10 based on criteria alignment"
    )
    reason_code: Literal["LOW_VOLUME", "MISSING_TRIGGER", "LOW_RISK_SCORE", "CHOPPY", "EXECUTED"] = Field(
        description="Strict machine-readable categorical token representing the core reason behind the action"
    )
    rationale: str = Field(
        description="Detailed technical justification and comprehensive market analysis supporting the final execution choice"
    )

class TradingAgent:
    def __init__(self, model_name: str):
        self.client = genai.Client()
        self.model_name = model_name

        self.system_prompt = (
            "You are an experienced S&P 500 intraday trader executing systematic, rule-based strategies.\n\n"
            "You receive pre-calculated market data:\n"
            "- refs: Key price levels (PDH, PDL, orbHigh, orbLow, vwapRTH, etc.)\n"
            "- stats: Market metrics (gap, atr5, rvol5, asiaChPct, euChPct, vixChPct, etc.)\n"
            "- bias: Directional bias flags (A/B/C/D) and position context (abovePDH, belowPDL, afterORB)\n"
            "- triggers: Entry signals (longBreak, longRetest, faderLong, reclaimLong, etc.)\n"
            "- risk: Pre-computed risk scores (riskScoreLong, riskScoreShort) and stops/targets\n\n"
            "YOUR JOB: Read the market structure, evaluate setup quality, and decide BUY/SELL/HOLD.\n\n"
            "ENTRY CRITERIA:\n"
            "- Strong setups: riskScore >= 7 AND at least one active trigger AND rvol5 > 1.0 -> action: BUY/SELL, position_size_pct: 100\n"
            "- Decent setups: riskScore 5-6 AND at least one active trigger AND rvol5 > 1.0 -> action: BUY/SELL, position_size_pct: 50\n"
            "- Weak/no setup: riskScore < 5 OR no active triggers OR rvol5 <= 1.0 -> action: HOLD, position_size_pct: 0\n\n"
            "BIAS CONTEXT:\n"
            "- Bias A (bull world + gap up): Favor LONG, avoid SHORT. Do not short unless riskScoreShort >= 8.\n"
            "- Bias B (bear world + gap down): Favor SHORT, avoid LONG. Do not long unless riskScoreLong >= 8.\n"
            "- Bias C (mean reversion): Fade extended moves. Max 50% position size.\n"
            "- Bias D (contrarian): Advanced reversal - needs strong confirmation.\n\n"
            "TRIGGERS (in priority order):\n"
            "1. Retest triggers (longRetest, shortRetest): Best quality - ORB break + pullback + vwap confirm.\n"
            "2. Breakout triggers (longBreak, shortBreak): Momentum plays - clean ORB violation.\n"
            "3. Reclaim triggers (reclaimLong, reclaimShort): Reversal plays - retaking key levels.\n"
            "4. Fader triggers (faderLong, faderShort): Counter-trend - max 50% position size.\n\n"
            "STOP/TARGET SELECTION (Only applicable if action is BUY or SELL):\n"
            "- Stop Loss: Compare longStopATR/shortStopATR with orbLow/orbHigh. Select the level that is closer to the current price (tighter stop).\n"
            "- If vixChPct > 10%: Widen the selected stop loss level by an additional 20% in price distance.\n"
            "- Targets: Use longTPATR/shortTPATR or 1R/2R levels based on optimal risk/reward.\n\n"
            "DECISION RULES:\n"
            "1. Never enter any trade if rvol5 <= 1.0.\n"
            "2. Counter-trend trades (faders, Bias C) are strictly capped at max 50% size.\n"
            "3. If market is choppy (orbWidthPct < 0.002, no active bias, rvol5 <= 1.0) -> FORCE HOLD.\n"
            "4. IF ACTION IS HOLD: You MUST set direction='NONE', position_size_pct=0, stop_loss=null, and take_profit=null.\n\n"
            "OUTPUT RATIONALE FORMAT:\n"
            "- If action is BUY/SELL: Bias [X] active. Trigger [Y] fired. Risk [score]. Setup quality: [high/medium/low]. [Any concerns].\n"
            "- If action is HOLD: Provide a comprehensive multi-variable analysis in the rationale field detailing structure, liquidity, and risk parameters. The description must explicitly quantify internal metrics (e.g., rvol5, orbWidthPct) to justify execution denial. Do not truncate the technical commentary.\n\n"
            "Think like a professional trader managing institutional risk. If the edge is not clear - HOLD."
        )

        self.user_template = (
            "Market Snapshot and Context:\n"
            "Symbol: {symbol}\n"
            "Date: {ny_date}\n\n"
            "Market Reference Levels:\n"
            "{refs}\n\n"
            "Statistical Indicators:\n"
            "{stats}\n\n"
            "Bias Matrix:\n"
            "{bias}\n\n"
            "Tactical Triggers:\n"
            "{triggers}\n\n"
            "Risk Matrix Pre-calculations:\n"
            "{risk}\n"
        )

    def execute_analysis(self, payload: MarketPayload) -> AgentDecision:
        meta = payload["meta"]

        user_message = self.user_template.format(
            symbol=meta["symbol"],
            ny_date=meta["nyDate"],
            refs=json.dumps(payload["refs"], indent=2),
            stats=json.dumps(payload["stats"], indent=2),
            bias=json.dumps(payload["bias"], indent=2),
            triggers=json.dumps(payload["triggers"], indent=2),
            risk=json.dumps(payload["risk"], indent=2)
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=AgentDecision,
            ),
        )

        return AgentDecision.model_validate_json(response.text)