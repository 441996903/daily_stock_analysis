#!/usr/bin/env python3
"""成交额龙头池计算：2024-01-02 起，每个交易日中证全A成交额第一入池，
连续89个交易日未再登顶踢出。

Token 从 stdin 第一行读入，只存内存，不落盘、不打印。
断点续跑：已算日期记在 daily_top1.jsonl，重跑自动跳过。
"""
import csv
import json
import os
import sys
import time
import urllib.request

API = "https://api.tushare.pro"
HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "20240101", "20261110"
KICK = 89

TOKEN = os.environ.get("TUSHARE_TOKEN", "").strip()
if not TOKEN and not sys.stdin.isatty():
    try:
        TOKEN = sys.stdin.readline().strip()
    except Exception:
        TOKEN = ""
if not TOKEN:
    print(json.dumps({"fatal": "no token (env TUSHARE_TOKEN / stdin)"}))
    sys.exit(2)


def call(api_name, params, fields, retries=3):
    payload = {"api_name": api_name, "token": TOKEN, "params": params, "fields": fields}
    body = json.dumps(payload).encode("utf-8")
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.loads(r.read().decode("utf-8"))
            if res.get("code") != 0:
                msg = str(res.get("msg"))
                if "IP" in msg and ("超限" in msg or "数量" in msg):
                    print(json.dumps({"fatal": f"IP_LIMIT: {msg}"}))
                    sys.exit(4)
                if "token" in msg.lower() or "Token" in msg:
                    print(json.dumps({"fatal": f"token rejected by tushare: {msg}"}))
                    sys.exit(3)
                raise RuntimeError(f"code={res.get('code')} msg={msg}")
            d = res.get("data") or {}
            cols = d.get("fields") or []
            return [dict(zip(cols, it)) for it in (d.get("items") or [])]
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{api_name} failed after retries: {last}")


def main():
    # 1) 交易日历（上交所口径，全A通用）
    cal = call("trade_cal", {"exchange": "SSE", "start_date": START, "end_date": END, "is_open": "1"},
               "cal_date,is_open")
    days = sorted(r["cal_date"] for r in cal)
    print(json.dumps({"stage": "cal", "trading_days": len(days), "first": days[0], "last": days[-1]}), flush=True)

    # 2) 股票名单（在市+退市+暂停，取名称/上市日，用于标注）
    info = {}
    for status in ("L", "D", "P"):
        for r in call("stock_basic", {"list_status": status}, "ts_code,name,list_date,delist_date"):
            info[r["ts_code"]] = r
    print(json.dumps({"stage": "stock_basic", "stocks": len(info)}), flush=True)

    # 3) 逐日成交额第一（断点续存）
    ckpt = os.path.join(HERE, "daily_top1.jsonl")
    top1 = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    top1[rec["date"]] = rec
    todo = [d for d in days if d not in top1]
    print(json.dumps({"stage": "daily", "todo": len(todo), "done": len(top1)}), flush=True)
    with open(ckpt, "a", encoding="utf-8") as f:
        for n, d in enumerate(todo, 1):
            rows = call("daily", {"trade_date": d}, "ts_code,amount")
            if not rows:
                rec = {"date": d, "ts_code": "", "amount": 0}
            else:
                best = max(rows, key=lambda r: (r.get("amount") or 0))
                rec = {"date": d, "ts_code": best["ts_code"], "amount": best.get("amount") or 0}
            top1[d] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0:
                print(json.dumps({"stage": "daily_progress", "done": n, "of": len(todo)}), flush=True)
            time.sleep(0.12)

    def name_of(code):
        return (info.get(code) or {}).get("name", "")

    def listed_days(code, date):
        ld = (info.get(code) or {}).get("list_date")
        if not ld:
            return None
        from datetime import date as _date
        a = _date(int(ld[:4]), int(ld[4:6]), int(ld[6:8]))
        b = _date(int(date[:4]), int(date[4:6]), int(date[6:8]))
        return (b - a).days

    # 4) 每日第一清单 CSV
    with open(os.path.join(HERE, "daily_top1.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "ts_code", "name", "amount_yi", "flags"])
        for d in days:
            rec = top1.get(d, {"ts_code": "", "amount": 0})
            code = rec["ts_code"]
            flags = []
            nm = name_of(code)
            if "ST" in nm:
                flags.append("ST")
            ld = listed_days(code, d) if code else None
            if ld is not None and ld < 90:
                flags.append(f"上市{ld}天")
            w.writerow([f"{d[:4]}-{d[4:6]}-{d[6:8]}", code, nm,
                        round((rec["amount"] or 0) / 100000, 2), ";".join(flags)])

    # 5) 池模拟：登顶续命，连续89个交易日未登顶踢出
    members = {}   # code -> last_top_idx
    ever_removed = set()
    events = []
    for idx, d in enumerate(days):
        code = top1.get(d, {}).get("ts_code", "")
        if code:
            if code not in members:
                events.append({"date": d, "action": "IN", "ts_code": code,
                               "kind": "RE" if code in ever_removed else "NEW"})
            members[code] = idx
        for c in list(members):
            if idx - members[c] >= KICK:
                events.append({"date": d, "action": "OUT", "ts_code": c, "kind": ""})
                ever_removed.add(c)
                del members[c]

    with open(os.path.join(HERE, "events.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "action", "ts_code", "name", "kind"])
        for e in events:
            w.writerow([f"{e['date'][:4]}-{e['date'][4:6]}-{e['date'][6:8]}", e["action"],
                        e["ts_code"], name_of(e["ts_code"]), e["kind"]])

    last_idx = len(days) - 1
    current = []
    for c, lt in sorted(members.items(), key=lambda kv: kv[1], reverse=True):
        since = last_idx - lt
        top_rec = top1[days[lt]]
        current.append({
            "ts_code": c, "name": name_of(c), "last_top_date": days[lt],
            "days_since": since, "days_left": KICK - since,
            "last_top_amount_yi": round((top_rec["amount"] or 0) / 100000, 2),
        })
    with open(os.path.join(HERE, "current_pool.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ts_code", "name", "last_top_date", "days_since", "days_left", "last_top_amount_yi"])
        for c in current:
            w.writerow([c["ts_code"], c["name"],
                        f"{c['last_top_date'][:4]}-{c['last_top_date'][4:6]}-{c['last_top_date'][6:8]}",
                        c["days_since"], c["days_left"], c["last_top_amount_yi"]])

    winners = {top1[d]["ts_code"] for d in days if top1.get(d, {}).get("ts_code")}
    summary = {
        "trading_days": len(days),
        "range": [days[0], days[-1]],
        "distinct_winners": len(winners),
        "events_in": sum(1 for e in events if e["action"] == "IN"),
        "events_out": sum(1 for e in events if e["action"] == "OUT"),
        "current_pool_size": len(current),
        "current_pool": current,
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({"stage": "DONE", **{k: v for k, v in summary.items() if k != "current_pool"}}), flush=True)


if __name__ == "__main__":
    main()
