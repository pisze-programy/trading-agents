from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import yfinance as yf
from pydantic import BaseModel, Field


class MarketMeta(BaseModel):
    symbol: str
    nyDate: str


class MarketBias(BaseModel):
    A: bool
    B: bool
    C: bool
    D: bool
    abovePDH: bool
    belowPDL: bool
    afterORB: bool


class MarketTriggers(BaseModel):
    longBreak: bool
    shortBreak: bool
    longRetest: bool
    shortRetest: bool
    faderLong: bool
    faderShort: bool
    reclaimLong: bool
    reclaimShort: bool


class MarketRisk(BaseModel):
    riskScoreLong: int
    riskScoreShort: int
    longStop: Optional[float] = None
    shortStop: Optional[float] = None
    long1R: Optional[float] = None
    long2R: Optional[float] = None
    short1R: Optional[float] = None
    short2R: Optional[float] = None
    longStopATR: Optional[float] = None
    longTPATR: Optional[float] = None
    shortStopATR: Optional[float] = None
    shortTPATR: Optional[float] = None
    slPts: Optional[float] = None
    tpPts: Optional[float] = None


class MarketPayload(BaseModel):
    meta: MarketMeta
    refs: Dict[str, float] = Field(description="Dynamic key-value pairs for non-null reference price levels like PDH, PDL, vwapRTH")
    stats: Dict[str, float] = Field(description="Dynamic key-value pairs for non-null statistical metrics like gap, atr5, rvol5")
    bias: MarketBias
    triggers: MarketTriggers
    risk: MarketRisk


class AnalyzeParams(BaseModel):
    symbol: str
    ny_date: str
    current_timestamp: float
    orb_minutes: Optional[int] = 5
    atr5_len: Optional[int] = 14
    atr5_sl_mult: Optional[float] = 1.2
    atr5_tp_mult: Optional[float] = 1.8
    phase: Optional[Literal["preUS", "postORB", "full"]] = "preUS"


class DataSource:
    def __init__(self):
        pass

    def get_rth_window(self, ny_date: str) -> Tuple[int, int]:
        dt = datetime.strptime(ny_date, "%Y-%m-%d")
        start = int(dt.replace(hour=15, minute=30, second=0).timestamp())
        end = int(dt.replace(hour=22, minute=0, second=0).timestamp())
        return start, end

    def get_orb_window(self, ny_date: str, orb_minutes: int) -> Tuple[int, int]:
        start, _ = self.get_rth_window(ny_date)
        end = start + (orb_minutes * 60)
        return start, end

    def get_on_window(self, ny_date: str) -> Tuple[int, int]:
        dt = datetime.strptime(ny_date, "%Y-%m-%d")
        prev_day = dt - timedelta(days=1)
        start = int(prev_day.replace(hour=18, minute=0, second=0).timestamp())
        end = int(dt.replace(hour=9, minute=30, second=0).timestamp())
        return start, end

    def get_asia_window(self, ny_date: str) -> Tuple[int, int]:
        dt = datetime.strptime(ny_date, "%Y-%m-%d")
        prev_day = dt - timedelta(days=1)
        start = int(prev_day.replace(hour=19, minute=0, second=0).timestamp())
        end = int(dt.replace(hour=4, minute=0, second=0).timestamp())
        return start, end

    def get_eu_window(self, ny_date: str) -> Tuple[int, int]:
        dt = datetime.strptime(ny_date, "%Y-%m-%d")
        start = int(dt.replace(hour=3, minute=0, second=0).timestamp())
        end = int(dt.replace(hour=11, minute=30, second=0).timestamp())
        return start, end

    def get_daily(self, symbol: str, ny_date: str, count: int) -> List[Dict[str, float]]:
        ticker = yf.Ticker(symbol)
        start_dt = datetime.strptime(ny_date, "%Y-%m-%d") - timedelta(days=count * 2)
        end_dt = datetime.strptime(ny_date, "%Y-%m-%d") + timedelta(days=1)

        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d"
        )

        bars = []
        for index, row in df.tail(count).iterrows():
            bars.append({
                "t": int(index.timestamp()),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row["Volume"])
            })
        return bars

    def get_intraday(self, symbol: str, ny_date: str, interval: str) -> List[Dict[str, Any]]:
        ticker = yf.Ticker(symbol)
        end_dt = datetime.strptime(ny_date, "%Y-%m-%d") + timedelta(days=1)

        df = ticker.history(
            start=ny_date,
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval
        )
        bars = []
        for index, row in df.iterrows():
            bars.append({
                "t": int(index.timestamp()),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row["Volume"])
            })
        return bars


def build_agent_payload(data_source: DataSource, params: AnalyzeParams) -> MarketPayload:
    rth_start, rth_end = data_source.get_rth_window(params.ny_date)
    orb_start, orb_end = data_source.get_orb_window(params.ny_date, params.orb_minutes)
    on_start, on_end = data_source.get_on_window(params.ny_date)
    asia_start, asia_end = data_source.get_asia_window(params.ny_date)
    eu_start, eu_end = data_source.get_eu_window(params.ny_date)

    cutoff_timestamp = params.current_timestamp

    daily = data_source.get_daily(params.symbol, params.ny_date, 60)
    m1_raw = data_source.get_intraday(params.symbol, params.ny_date, "1m")
    m5_raw = data_source.get_intraday(params.symbol, params.ny_date, "5m")

    m1 = [b for b in m1_raw if b["t"] <= cutoff_timestamp]

    m5_historical = [b for b in m5_raw if b["t"] <= cutoff_timestamp]
    m5 = m5_historical[-(params.atr5_len + 10):]

    vix = data_source.get_daily("^VIX", params.ny_date, 2)
    dxy = data_source.get_daily("DX-Y.NYB", params.ny_date, 2)
    us10 = data_source.get_daily("^TNX", params.ny_date, 2)

    rth_bars_m1 = [b for b in m1 if rth_start <= b["t"] <= min(rth_end, cutoff_timestamp)]
    on_bars_m1 = [b for b in m1 if on_start <= b["t"] <= min(on_end, cutoff_timestamp)]
    asia_bars = [b for b in m1 if asia_start <= b["t"] <= min(asia_end, cutoff_timestamp)]
    eu_bars = [b for b in m1 if eu_start <= b["t"] <= min(eu_end, cutoff_timestamp)]
    orb_bars = [b for b in m1 if orb_start <= b["t"] <= min(orb_end, cutoff_timestamp)]

    def get_extrema(bars: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        if not bars:
            return {"hi": None, "lo": None}
        return {"hi": max(b["h"] for b in bars), "lo": min(b["l"] for b in bars)}

    rth_x = get_extrema(rth_bars_m1)
    on_x = get_extrema(on_bars_m1)
    orb_x = get_extrema(orb_bars)

    prev_daily = daily[-2] if len(daily) >= 2 else None
    prev_close = prev_daily["c"] if prev_daily else None
    pdh = prev_daily["h"] if prev_daily else None
    pdl = prev_daily["l"] if prev_daily else None

    def calculate_anchored_vwap(source_bars: List[Dict[str, Any]], anchor_bar: Dict[str, Any]) -> Optional[float]:
        try:
            start_index = source_bars.index(anchor_bar)
            active_bars = source_bars[start_index:]
            total_pv = sum(b["c"] * b["v"] for b in active_bars)
            total_v = sum(b["v"] for b in active_bars)
            return total_pv / total_v if total_v > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    vwap_on = calculate_anchored_vwap(m1, on_bars_m1[0]) if on_bars_m1 else None
    vwap_rth = calculate_anchored_vwap(m1, rth_bars_m1[0]) if rth_bars_m1 else None

    def calculate_atr(bars: List[Dict[str, Any]], period: int) -> Optional[float]:
        if len(bars) < period:
            return None
        tr_values = []
        for i in range(1, len(bars)):
            high = bars[i]["h"]
            low = bars[i]["l"]
            prev_close_val = bars[i - 1]["c"]
            tr = max(high - low, abs(high - prev_close_val), abs(low - prev_close_val))
            tr_values.append(tr)
        return sum(tr_values[-period:]) / period if tr_values else None

    atr_d20 = calculate_atr(daily, 20)

    rth_open = rth_bars_m1[0]["o"] if rth_bars_m1 else None
    gap = (rth_open - prev_close) / prev_close if prev_close and rth_open else None
    gap_ratio = abs(gap) / (atr_d20 / rth_open) if gap is not None and atr_d20 and rth_open else None

    asia_open = asia_bars[0]["o"] if asia_bars else None
    asia_close = asia_bars[-1]["c"] if asia_bars else None
    eu_open = eu_bars[0]["o"] if eu_bars else None
    eu_close = eu_bars[-1]["c"] if eu_bars else None

    asia_ch_pct = (asia_close - asia_open) / asia_open if asia_close is not None and asia_open else None
    eu_ch_pct = (eu_close - eu_open) / eu_open if eu_close is not None and eu_open else None

    orb_width = orb_x["hi"] - orb_x["lo"] if orb_x["hi"] is not None and orb_x["lo"] is not None else None
    orb_width_pct = orb_width / rth_open if orb_width is not None and rth_open else None

    bull_world = (asia_ch_pct or 0.0) > 0.0 and (eu_ch_pct or 0.0) > 0.0
    bear_world = (asia_ch_pct or 0.0) < 0.0 and (eu_ch_pct or 0.0) < 0.0
    big_gap_up = gap_ratio is not None and (gap or 0.0) > 0.0 and gap_ratio >= 0.6
    big_gap_dn = gap_ratio is not None and (gap or 0.0) < 0.0 and gap_ratio >= 0.6
    mid_gap = gap_ratio is not None and 0.2 <= gap_ratio < 0.6

    last_bar = rth_bars_m1[-1] if rth_bars_m1 else None
    above_pdh = pdh is not None and last_bar is not None and last_bar["c"] > pdh
    below_pdl = pdl is not None and last_bar is not None and last_bar["c"] < pdl

    after_orb = cutoff_timestamp >= orb_end

    bias_a = bool(bull_world and big_gap_up and (above_pdh or (rth_open and pdh and rth_open > pdh)))
    bias_b = bool(bear_world and big_gap_dn and (below_pdl or (rth_open and pdl and rth_open < pdl)))

    bias_c = False
    if mid_gap and last_bar is not None:
        if (gap or 0.0) > 0.0:
            bias_c = last_bar["c"] < (vwap_rth if vwap_rth is not None else float("inf"))
        else:
            bias_c = last_bar["c"] > (vwap_rth if vwap_rth is not None else float("-inf"))

    overnight_bias_up = prev_close is not None and vwap_on is not None and vwap_on > prev_close
    overnight_bias_dn = prev_close is not None and vwap_on is not None and vwap_on < prev_close
    bias_d = bool((overnight_bias_up and after_orb and below_pdl) or (overnight_bias_dn and after_orb and above_pdh))

    long_break = bool(after_orb and last_bar and orb_x["hi"] is not None and last_bar["c"] > orb_x["hi"])
    short_break = bool(after_orb and last_bar and orb_x["lo"] is not None and last_bar["c"] < orb_x["lo"])

    long_retest = bool(
        after_orb and last_bar and orb_x["hi"] is not None and vwap_rth is not None and last_bar["c"] > orb_x["hi"] and
        last_bar["l"] <= orb_x["hi"] and last_bar["c"] > vwap_rth)
    short_retest = bool(
        after_orb and last_bar and orb_x["lo"] is not None and vwap_rth is not None and last_bar["c"] < orb_x["lo"] and
        last_bar["h"] >= orb_x["lo"] and last_bar["c"] < vwap_rth)

    cross_vwap_rth = bool(after_orb and vwap_rth is not None and last_bar and last_bar["c"] > vwap_rth)
    under_vwap_rth = bool(after_orb and vwap_rth is not None and last_bar and last_bar["c"] < vwap_rth)
    fader_long = bool(after_orb and mid_gap and (gap or 0.0) < 0.0 and cross_vwap_rth)
    fader_short = bool(after_orb and mid_gap and (gap or 0.0) > 0.0 and under_vwap_rth)

    recent_bars = rth_bars_m1[-20:] if rth_bars_m1 else []
    was_below_pdh_recent = pdh is not None and any(b["c"] <= pdh for b in recent_bars)
    was_above_pdl_recent = pdl is not None and any(b["c"] >= pdl for b in recent_bars)
    reclaim_long = bool(after_orb and above_pdh and was_below_pdh_recent)
    reclaim_short = bool(after_orb and below_pdl and was_above_pdl_recent)

    long_stop = orb_x["lo"]
    short_stop = orb_x["hi"]
    risk_pts_long = last_bar["c"] - long_stop if last_bar and long_stop is not None else None
    risk_pts_short = short_stop - last_bar["c"] if last_bar and short_stop is not None else None

    long_1r = last_bar["c"] + risk_pts_long if risk_pts_long is not None and last_bar else None
    long_2r = last_bar["c"] + 2 * risk_pts_long if risk_pts_long is not None and last_bar else None
    short_1r = last_bar["c"] - risk_pts_short if risk_pts_short is not None and last_bar else None
    short_2r = last_bar["c"] - 2 * risk_pts_short if risk_pts_short is not None and last_bar else None

    atr5 = calculate_atr(m5, params.atr5_len)
    vol5 = m5[-1]["v"] if m5 else None

    volumes = [b["v"] for b in m5]
    vol5_sma20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    rvol5 = vol5 / vol5_sma20 if vol5 and vol5_sma20 else None

    sl_pts = atr5 * params.atr5_sl_mult if atr5 else None
    tp_pts = atr5 * params.atr5_tp_mult if atr5 else None

    long_stop_atr = last_bar["c"] - sl_pts if last_bar and sl_pts is not None else None
    long_tp_atr = last_bar["c"] + tp_pts if last_bar and tp_pts is not None else None
    short_stop_atr = last_bar["c"] + sl_pts if last_bar and sl_pts is not None else None
    short_tp_atr = last_bar["c"] - tp_pts if last_bar and tp_pts is not None else None

    orb_ok = orb_width_pct is not None and 0.001 < orb_width_pct < 0.006
    base_long = (2 if bull_world else 0) + (2 if big_gap_up else (1 if mid_gap else 0)) + (1 if orb_ok else 0)
    base_short = (2 if bear_world else 0) + (2 if big_gap_dn else (1 if mid_gap else 0)) + (1 if orb_ok else 0)
    mom_long = (2 if last_bar and vwap_rth and last_bar["c"] > vwap_rth else 0) + (1 if above_pdh else 0) + (
        1 if after_orb and long_break else 0)
    mom_short = (2 if last_bar and vwap_rth and last_bar["c"] < vwap_rth else 0) + (1 if below_pdl else 0) + (
        1 if after_orb and short_break else 0)
    risk_score_long = min(10, base_long + mom_long)
    risk_score_short = min(10, base_short + mom_short)

    def calculate_macro_pct(macro_data: List[Dict[str, Any]]) -> Optional[float]:
        if len(macro_data) < 2:
            return None
        return (macro_data[-1]["c"] - macro_data[-2]["c"]) / macro_data[-2]["c"]

    vix_ch_pct = calculate_macro_pct(vix)
    dxy_ch_pct = calculate_macro_pct(dxy)
    us10y_ch_pct = calculate_macro_pct(us10)

    raw_refs = {
        "PDH": pdh, "PDL": pdl, "PrevClose": prev_close, "RTHH": rth_x["hi"], "RTHL": rth_x["lo"],
        "ONH": on_x["hi"], "ONL": on_x["lo"], "vwapON": vwap_on, "vwapRTH": vwap_rth,
        "orbHigh": orb_x["hi"], "orbLow": orb_x["lo"], "orbWidth": orb_width, "orbWidthPct": orb_width_pct
    }
    refs = {k: v for k, v in raw_refs.items() if v is not None}

    raw_stats = {
        "atrD20": atr_d20, "gap": gap, "gapRatio": gap_ratio, "asiaChPct": asia_ch_pct, "euChPct": eu_ch_pct,
        "atr5": atr5, "rvol5": rvol5, "vixChPct": vix_ch_pct, "dxyChPct": dxy_ch_pct, "us10yChPct": us10y_ch_pct
    }
    stats = {k: v for k, v in raw_stats.items() if v is not None}

    return MarketPayload(
        meta=MarketMeta(symbol=params.symbol, nyDate=params.ny_date),
        refs=refs,
        stats=stats,
        bias=MarketBias(
            A=bias_a, B=bias_b, C=bias_c, D=bias_d,
            abovePDH=above_pdh, belowPDL=below_pdl, afterORB=after_orb
        ),
        triggers=MarketTriggers(
            longBreak=long_break, shortBreak=short_break,
            longRetest=long_retest, shortRetest=short_retest,
            faderLong=fader_long, faderShort=fader_short,
            reclaimLong=reclaim_long, reclaimShort=reclaim_short
        ),
        risk=MarketRisk(
            riskScoreLong=risk_score_long, riskScoreShort=risk_score_short,
            longStop=long_stop, shortStop=short_stop, long1R=long_1r, long2R=long_2r,
            short1R=short_1r, short2R=short_2r, longStopATR=long_stop_atr, longTPATR=long_tp_atr,
            shortStopATR=short_stop_atr, shortTPATR=short_tp_atr, slPts=sl_pts, tpPts=tp_pts
        )
    )