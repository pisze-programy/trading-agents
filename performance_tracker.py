class PerformanceTracker:
    def __init__(self):
        self.history = []

    def update(self, positions: dict, current_price: float):
        for name, pos in positions.items():
            status = getattr(pos, "status", None) or pos.get("status")
            if status == "ACTIVE":
                direction = getattr(pos, "direction", None) or pos.get("direction")
                stop_loss = getattr(pos, "stop_loss", None) or pos.get("stop_loss")
                take_profit = getattr(pos, "take_profit", None) or pos.get("take_profit")
                entry_price = getattr(pos, "entry_price", None) or pos.get("entry_price")

                if direction == "SHORT" and current_price >= stop_loss:
                    self.set_status_idle(pos)
                    self.record(name, direction, entry_price, stop_loss)
                elif direction == "SHORT" and current_price <= take_profit:
                    self.set_status_idle(pos)
                    self.record(name, direction, entry_price, take_profit)

    def record(self, name: str, direction: str, entry: float, exit: float):
        pnl = entry - exit if direction == "SHORT" else exit - entry
        self.history.append({
            "name": name,
            "pnl": pnl,
            "win": pnl > 0
        })
        self.print_stats()

    def print_stats(self):
        stats = {}
        for trade in self.history:
            name = trade["name"]
            if name not in stats:
                stats[name] = {"trades": 0, "wins": 0, "pnl": 0.0}
            stats[name]["trades"] += 1
            stats[name]["pnl"] += trade["pnl"]
            if trade["win"]:
                stats[name]["wins"] += 1

        print("\n=== PERFORMANCE SUMMARY ===")
        for name, data in stats.items():
            wr = (data["wins"] / data["trades"]) * 100
            print(f"{name} | Trades: {data['trades']} | WinRate: {wr:.1f}% | Total PnL: {data['pnl']:.2f}")