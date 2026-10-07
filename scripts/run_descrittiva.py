"""Fase 1 - Analisi descrittiva: qualità del dato, andamenti per gruppo, correlazioni, rendimenti.

Uso: python scripts/run_descrittiva.py [--html]
Con --html salva i grafici in reports/figures/.
"""

import argparse

from cripto_trend import plots
from cripto_trend.config import DATETIME, REPORTS_DIR
from cripto_trend.data import GROUPS, load_raw, clean, rename_close_columns
from cripto_trend.features import returns


def main(save_html: bool) -> None:
    raw = load_raw()
    df = clean(raw)
    print(f"Righe: {len(raw)} -> {len(df)} dopo la pulizia | colonne: {raw.shape[1]}")
    print(f"Periodo: {df[DATETIME].min()} -> {df[DATETIME].max()} "
          f"({(df[DATETIME].max() - df[DATETIME].min()).days} giorni)")

    close = rename_close_columns(df)
    corr_btc = close.drop(columns=DATETIME).corr()["BTC"].drop("BTC").sort_values()
    print("\nCorrelazione con BTC (più negative e più positive):")
    print(corr_btc.head(5).round(2).to_string(), "\n...\n", corr_btc.tail(5).round(2).to_string())

    rend = returns(close, ["BTC", "SP", "NASDAQ", "gold"])
    print("\nRendimento medio a 1 anno:")
    print(rend.filter(like="_1y").mean().mul(100).round(1).to_string())

    if save_html:
        out = REPORTS_DIR / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for name, cols in GROUPS.items():
            plots.group_vs_btc(close, cols, name).write_html(out / f"gruppo_{name.replace(' ', '_')}.html")
        plots.correlation_heatmap(close.drop(columns=DATETIME)).write_html(out / "correlazioni.html")
        print(f"\nGrafici salvati in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", action="store_true", help="salva i grafici in reports/figures")
    main(parser.parse_args().html)
