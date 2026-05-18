import time

from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing_extensions import TypedDict


class TradingSession(BaseModel):
    name: str


class AgentState(TypedDict):
    name: str


def hello_world() -> dict:
    # Fetch data from yfinance
    return {"sp500": 5100.25, "is_close_window": False}


workflow = StateGraph(AgentState)

workflow.add_node("hello_world", hello_world)
workflow.add_edge("hello_world", END)

arena_trader = workflow.compile()

if __name__ == "__main__":
    session_state = {
        "agents_sessions": {
            "gemini": TradingSession(),
        }
    }

    while True:
        # Fetch data every 1min
        time.sleep(60)