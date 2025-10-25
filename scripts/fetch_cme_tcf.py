import pandas as pd, requests, io, os, datetime as dt

OUT = "data/raw"  # repo root relative
SYMBOLS = {"CL":"NYMEX WTI Crude Oil", "NG":"Henry Hub NatGas"}  # for filtering
START, END = dt.date(2023,12,9), dt.date.today()  # earliest visible on FTP

def url_for(d): return f"https://www.cmegroup.com/ftp/settle/TCF/TCF_{d:%Y%m%d}.csv"

os.makedirs(OUT, exist_ok=True)
for sym in SYMBOLS:
    os.makedirs(f"{OUT}/{sym}", exist_ok=True)

d = START
while d <= END:
    url = url_for(d)
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or not r.text.strip():
            d += dt.timedelta(days=1); continue
        df = pd.read_csv(io.StringIO(r.text))
        # Inspect one file to confirm exact column names; adjust these if needed:
        # Typical columns include product/contract and settlement fields.
        cols = {c.lower().strip(): c for c in df.columns}
        # commonly seen names (adjust after inspecting a sample CSV):
        prod = cols.get("product", cols.get("productcode", cols.get("prodcode", None)))
        month = cols.get("month", cols.get("contractmonth", None))
        year  = cols.get("year", cols.get("contractyear", None))
        settle= cols.get("settle", cols.get("settlement price", cols.get("settlement", None)))
        vol   = cols.get("volume", None)
        oi    = cols.get("openinterest", cols.get("open interest", None))

        if not all([prod, month, year, settle]):
            print(f"⚠️ columns unexpected for {d}: {df.columns.tolist()}"); d += dt.timedelta(days=1); continue

        df = df[[prod, month, year, settle] + [x for x in [vol, oi] if x]]
        df.columns = ["product","month","year","settle"] + ([ "volume" ] if vol else []) + ([ "open_interest" ] if oi else [])
        df["date"] = d

        # keep CL/NG only
        keep = df["product"].astype(str).str.upper().isin(["CL","NG"])
        df = df[keep].copy()
        if df.empty:
            d += dt.timedelta(days=1); continue

        # to your raw schema
        month_map = dict(zip(list("FGHJKMNQUVXZ"), range(1,13)))
        def expiry_from_row(r):
            m = r["month"].strip().upper()[0]
            y = int(str(r["year"]).strip()[-2:])
            yyyy = 2000 + y if y < 80 else 1900 + y
            mm = month_map.get(m, None)
            # use 1st of contract month as expiry placeholder; your loader will refine later
            return dt.date(yyyy, mm, 1) if mm else pd.NaT

        out_rows = []
        for _, r in df.iterrows():
            expiry = expiry_from_row(r)
            out_rows.append({
                "date": r["date"],
                "symbol": r["product"].upper(),
                "expiry": expiry,
                "settle": float(r["settle"]),
                "last": None, "bid": None, "ask": None,
                "volume": int(r["volume"]) if "volume" in df.columns and pd.notnull(r["volume"]) else None,
                "open_interest": int(r["open_interest"]) if "open_interest" in df.columns and pd.notnull(r["open_interest"]) else None,
            })
        out = pd.DataFrame(out_rows).sort_values(["date","expiry"])
        if not out.empty:
            out.to_csv(f"{OUT}/{out.iloc[0]['symbol']}/settles_{d:%Y%m%d}.csv", index=False)
            print(f"saved {out.iloc[0]['symbol']} {d}")
    except Exception as e:
        print("skip", d, e)
    d += dt.timedelta(days=1)
