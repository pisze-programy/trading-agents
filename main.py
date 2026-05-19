import time
from datetime import datetime
from typing import Literal, Optional, Dict
from zoneinfo import ZoneInfo

from langgraph.graph import StateGraph, END
from pydantic import Field, BaseModel
from typing_extensions import TypedDict

from payload_builder import DataSource, MarketPayload, build_agent_payload, AnalyzeParams
from performance_tracker import PerformanceTracker
from trading_agent import AgentDecision, TradingAgent


class PositionState(BaseModel):
    status: Literal["IDLE", "ACTIVE", "FINISHED"] = Field(default="IDLE")
    direction: Literal["LONG", "SHORT", "NONE"] = Field(default="NONE")
    entry_price: Optional[float] = Field(default=None)
    entry_timestamp: Optional[float] = Field(default=None)
    stop_loss: Optional[float] = Field(default=None)
    take_profit: Optional[float] = Field(default=None)


class AgentState(TypedDict):
    symbol: str
    ny_date: str
    current_timestamp: float
    market_payload: Optional[MarketPayload]
    positions: Dict[str, PositionState]
    latest_decisions: Dict[str, dict]


data_source = DataSource()
trading_agent = TradingAgent(model_name="gemini-2.5-flash")

active_models = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-3.1-pro"
]

def data_ingestion_node(state: AgentState) -> dict:
    current_ts = state["current_timestamp"]
    ny_date = state["ny_date"]

    rth_start, rth_end = data_source.get_rth_window(ny_date)
    _, orb_end = data_source.get_orb_window(ny_date, orb_minutes=30)

    print(f"DEBUG TIME: Current TS: {current_ts} | ORB End TS: {orb_end} | RTH End TS: {rth_end}")

    if current_ts < orb_end:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingestion skipped: Before ORB session end")
        return {"market_payload": None}

    if current_ts > rth_end:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingestion skipped: Post-market closing time")
        return {"market_payload": None}

    params = AnalyzeParams(
        symbol=state["symbol"],
        ny_date=state["ny_date"],
        phase="postORB",
        current_timestamp=current_ts,
        orb_minutes=30,
        atr5_len=20,
        atr5_sl_mult=2.0,
        atr5_tp_mult=4.0
    )

    payload: MarketPayload = build_agent_payload(data_source, params)
    return {"market_payload": payload.model_dump()}


def agent_decision_node(state: AgentState) -> dict:
    payload_dict = state["market_payload"]
    if not payload_dict:
        return {}

    updated_positions = {k: v.model_copy() for k, v in state["positions"].items()}
    updated_decisions = {}

    for agent_id in active_models:
        position = updated_positions[agent_id]

        decision_pydantic: AgentDecision = trading_agent.execute_analysis(payload_dict)
        decision = decision_pydantic.model_dump()
        updated_decisions[agent_id] = decision

        current_close = payload_dict["refs"].get("PrevClose")

        if position.status == "IDLE":
            if decision["action"] in ["BUY", "SELL"]:
                position.status = "ACTIVE"
                position.direction = "LONG" if decision["action"] == "BUY" else "SHORT"
                position.entry_price = current_close
                position.entry_timestamp = state["current_timestamp"]
                position.stop_loss = decision["stop_loss"]
                position.take_profit = decision["take_profit"]

        elif position.status == "ACTIVE":
            time_elapsed = state["current_timestamp"] - position.entry_timestamp
            timeout_triggered = time_elapsed >= 7200

            if decision["action"] == "CLOSE" or timeout_triggered:
                position.status = "FINISHED"
                position.direction = "NONE"
                position.entry_price = None
                position.entry_timestamp = None
                position.stop_loss = None
                position.take_profit = None

    return {
        "positions": updated_positions,
        "latest_decisions": updated_decisions
    }


def check_positions_node(state: AgentState) -> dict:
    tracker = PerformanceTracker()
    current_price = state["market_payload"]["refs"]["vwapRTH"]
    tracker.update(state["positions"], current_price)
    return {}


workflow = StateGraph(AgentState)
workflow.add_node("data_ingestion", data_ingestion_node)
workflow.add_node("agent_decision", agent_decision_node)
workflow.add_node("check_positions", check_positions_node)

workflow.set_entry_point("data_ingestion")
workflow.add_edge("data_ingestion", "agent_decision")
workflow.add_edge("agent_decision", "check_positions")
workflow.add_edge("check_positions", END)

arena_trader = workflow.compile()

if __name__ == "__main__":

    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)

    current_state = {
        "symbol": "SPY",
        "ny_date": now_ny.strftime("%Y-%m-%d"),
        "current_timestamp": time.time(),
        "market_payload": None,
        "positions": {model_id: PositionState() for model_id in active_models},
        "latest_decisions": {}
    }

    print(f"Starting Live Trading Graph Loop for {current_state['symbol']}...")
    print(f"Current NY Market Date: {current_state['ny_date']}")

    while True:
        current_state["current_timestamp"] = time.time()

        final_output = arena_trader.invoke(current_state)

        current_state["positions"] = final_output["positions"]
        current_state["latest_decisions"] = final_output["latest_decisions"]

        for model_id in active_models:
            pos = current_state["positions"][model_id]
            dec = current_state["latest_decisions"].get(model_id, {})
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Model: {model_id} | Status: {pos.status} | Last Action: {dec.get('action', 'NONE')}")

        time.sleep(60)