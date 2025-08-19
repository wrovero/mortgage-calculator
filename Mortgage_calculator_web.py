# streamlit_app.py
# Run with:  streamlit run streamlit_app.py
# Requires:  pip install streamlit pandas matplotlib openpyxl
# Run application in the terminal with: streamlit run "c:\Users\wagne\OneDrive\Documents\Projects\Coding\Python\Learning\Mortgage\Mortgage_calculator_web.py"

import io
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Core engine (lifted from your CLI version, trimmed for app use) ----------


@dataclass
class Installment:
    number: int
    base_payment: float
    overpay: float
    lump: float
    interest: float
    principal: float
    balance: float
    cashback: float
    net_paid: float
    rate_annual_pct: float


def annuity_payment(balance: float, annual_rate_pct: float, periods: int, payments_per_year: int) -> float:
    if periods <= 0:
        return balance
    r = (annual_rate_pct / 100.0) / payments_per_year
    if abs(r) < 1e-15:
        return balance / periods
    return balance * r / (1 - (1 + r) ** (-periods))


def build_schedule(
    principal: float,
    payments_per_year: int,
    years: int,
    base_annual_rate_pct: float,
    # Overpayments
    regular_overpay: float = 0.0,
    overpay_start_period: int = 1,
    strategy: Literal["reduce_term", "reduce_payment"] = "reduce_term",
    # Lump sums + variable rates + special periods
    lump_sums: Optional[Dict[int, float]] = None,
    rate_changes: Optional[Dict[int, float]] = None,
    interest_only_windows: Optional[List[Tuple[int, int]]] = None,
    part_and_part_amount: float = 0.0,
    part_and_part_window: Optional[Tuple[int, int]] = None,
    # Cashbacks
    drawdown_cashback_fixed: float = 0.0,
    drawdown_cashback_pct: float = 0.0,
    recurring_cashback_first_n: int = 0,
    recurring_cashback_amount: float = 0.0,
    cashback_period_amounts: Optional[Dict[int, float]] = None,
    round_cents: bool = True,
) -> List[Installment]:

    total_periods = years * payments_per_year

    def rnd(x: float) -> float:
        return round(x + 1e-12, 2) if round_cents else x

    def period_in_ranges(p: int, ranges: List[Tuple[int, int]]) -> bool:
        return any(a <= p <= b for (a, b) in ranges)

    lump_sums = lump_sums or {}
    rate_changes = rate_changes or {}
    interest_only_windows = interest_only_windows or []
    cashback_period_amounts = cashback_period_amounts or {}

    if 1 not in rate_changes:
        rate_changes[1] = base_annual_rate_pct

    repay_bal = principal
    io_bal = 0.0

    # part-and-part pre-seed
    pnp_active = False
    if part_and_part_amount > 0 and part_and_part_window:
        start_pp, end_pp = part_and_part_window
        if start_pp <= 1 <= end_pp:
            move = min(part_and_part_amount, repay_bal)
            repay_bal -= move
            io_bal += move
            pnp_active = True

    current_rate = rate_changes[max(k for k in rate_changes.keys() if k <= 1)]
    base_payment = annuity_payment(
        repay_bal, current_rate, total_periods, payments_per_year)

    schedule: List[Installment] = []

    # Upfront cashback row (period 0) for reporting
    upfront_cashback = rnd(drawdown_cashback_fixed +
                           principal * (drawdown_cashback_pct / 100.0))
    if upfront_cashback < 0:
        upfront_cashback = 0.0

    k = 0
    while (repay_bal + io_bal) > 1e-8 and (strategy == "reduce_term" or k < total_periods):
        k += 1

        if k in rate_changes:
            current_rate = rate_changes[k]
            remaining = max(total_periods - (k - 1), 1)
            base_payment = annuity_payment(
                repay_bal, current_rate, remaining, payments_per_year)

        if part_and_part_amount > 0 and part_and_part_window:
            start_pp, end_pp = part_and_part_window
            if k == start_pp and not pnp_active:
                move = min(part_and_part_amount, repay_bal)
                repay_bal -= move
                io_bal += move
                pnp_active = True
                remaining = max(total_periods - (k - 1), 1)
                base_payment = annuity_payment(
                    repay_bal, current_rate, remaining, payments_per_year)

        is_full_io = period_in_ranges(k, interest_only_windows)

        period_rate = (current_rate / 100.0) / payments_per_year
        opening_total = repay_bal + io_bal
        interest = opening_total * period_rate

        this_overpay = regular_overpay if k >= max(
            1, overpay_start_period) else 0.0
        this_lump = (lump_sums or {}).get(k, 0.0)
        scheduled_base = interest if is_full_io else base_payment

        # Cashbacks this period
        cb_flat = recurring_cashback_amount if (
            recurring_cashback_first_n > 0 and k <= recurring_cashback_first_n) else 0.0
        cb_map = (cashback_period_amounts or {}).get(k, 0.0)
        this_cashback = max(0.0, cb_flat + cb_map)

        # Outflow/inflow math
        cash_out = scheduled_base + this_overpay + this_lump
        principal_available = max(0.0, cash_out - interest)

        # allocate principal
        principal_to_repay = min(repay_bal, principal_available)
        repay_bal -= principal_to_repay
        principal_available -= principal_to_repay

        principal_to_io = min(io_bal, principal_available)
        io_bal -= principal_to_io
        principal_available -= principal_to_io

        principal_reduction = principal_to_repay + principal_to_io
        new_total = repay_bal + io_bal

        # end of part-and-part window: roll IO into repay
        if part_and_part_amount > 0 and part_and_part_window:
            start_pp, end_pp = part_and_part_window
            if k == end_pp and pnp_active and io_bal > 1e-12:
                repay_bal += io_bal
                io_bal = 0.0
                pnp_active = False
                if repay_bal > 1e-8:
                    remaining = max(total_periods - k, 1)
                    base_payment = annuity_payment(
                        repay_bal, current_rate, remaining, payments_per_year)

        if strategy == "reduce_payment" and new_total > 1e-8:
            remaining = max(total_periods - k, 1)
            base_payment = annuity_payment(
                repay_bal, current_rate, remaining, payments_per_year)

        base_out = round(scheduled_base, 2)
        over_out = round(this_overpay, 2)
        lump_out = round(this_lump, 2)
        int_out = round(interest, 2)
        prin_out = round(principal_reduction, 2)
        bal_out = round(new_total, 2)
        cb_out = round(this_cashback, 2)
        net_paid_out = round(base_out + over_out + lump_out - cb_out, 2)

        is_last = (new_total <= 1e-8) or (strategy ==
                                          "reduce_payment" and k == total_periods)
        if is_last:
            bal_out = 0.0

        schedule.append(Installment(k, base_out, over_out, lump_out, int_out, prin_out, bal_out, cb_out, net_paid_out, round(current_rate, 4)
                                    ))

        if k > total_periods * 3 + 10_000:
            raise RuntimeError("Schedule did not converge—check inputs.")

    if upfront_cashback > 0:
        schedule.insert(0, Installment(0, 0.0, 0.0, 0.0, 0.0, 0.0, round(
            principal, 2), upfront_cashback, -upfront_cashback, round(current_rate, 4)))

    return schedule


def totals(schedule: List[Installment]) -> dict:
    total_payment = sum(i.base_payment + i.overpay + i.lump for i in schedule)
    total_interest = sum(i.interest for i in schedule)
    total_principal = sum(i.principal for i in schedule)
    total_cashback = sum(i.cashback for i in schedule)
    net_total_paid = round(total_payment - total_cashback, 2)
    return {
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "total_principal": round(total_principal, 2),
        "total_cashback": round(total_cashback, 2),
        "net_total_paid": net_total_paid,
        "num_installments": len([r for r in schedule if r.number > 0]),
    }

# ---------- Streamlit UI helpers ----------


def parse_period_amount_map(text: str) -> Dict[int, float]:
    text = (text or "").strip()
    if not text:
        return {}
    out: Dict[int, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        k, v = token.split(":")
        out[int(k.strip())] = float(v.strip())
    return out


def parse_period_rate_map(text: str) -> Dict[int, float]:
    text = (text or "").strip()
    if not text:
        return {}
    out: Dict[int, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        k, v = token.split(":")
        out[int(k.strip())] = float(v.strip().rstrip("%"))
    return out


def parse_ranges(text: str) -> List[Tuple[int, int]]:
    text = (text or "").strip()
    if not text:
        return []
    out: List[Tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        a, b = token.split("-")
        out.append((int(a.strip()), int(b.strip())))
    return out

# ---------- Streamlit App ----------


st.set_page_config(page_title="Irish Mortgage Calculator", layout="wide")
st.title("🇮🇪 Irish Mortgage Calculator – Web App")
st.caption(
    "Annuity schedule with overpayments, rate changes, IO windows, part-and-part, and cashbacks.")

with st.sidebar:
    st.header("Loan Basics")
    principal = st.number_input(
        "Mortgage amount (€)", min_value=0.0, value=400000.0, step=1000.0)
    rate = st.number_input("Starting nominal annual rate (%)",
                           min_value=0.0, value=3.40, step=0.05)
    term_years = st.number_input("Term (years)", min_value=1, value=35, step=1)
    freq = st.selectbox("Repayment frequency", options=[("Monthly", 12), ("Fortnightly", 26), (
        "Weekly", 52), ("Quarterly", 4)], index=0, format_func=lambda x: x[0])
    payments_per_year = freq[1]

    st.header("Rate Type")
    rate_type = st.selectbox("Rate type", options=[
                             "Fixed", "Variable"], index=0)
    fixed_years = st.number_input("Fixed period (years)", min_value=1, max_value=int(
        term_years), value=1, step=1, disabled=(rate_type == "Variable"))
    roll_rate_after_fixed = st.number_input(
        "Roll rate after fixed period (%)", min_value=0.0, value=rate, step=0.05, disabled=(rate_type == "Variable"))
    rate_text = f"{(fixed_years * payments_per_year) + 1}:{roll_rate_after_fixed}" if rate_type == "Fixed" else ""
    # rate_text = st.text_input(
    #     "Variable rates (Starting installment # and new variable rate %, e.g. 61:4.15%)", value="")
    st.header("Overpayments")
    regular_overpay = st.number_input(
        "Regular overpayment per installment (€)", min_value=0.0, value=0.0, step=50.0)
    overpay_start = st.number_input(
        "Start overpaying from installment #", min_value=1, value=1, step=1)
    strategy = st.selectbox("Overpayment strategy", options=[("Reduce term", "reduce_term"), (
        "Reduce payment", "reduce_payment")], index=0, format_func=lambda x: x[0])[1]

    st.header("Extras")
    lump_text = st.text_input(
        "One-off lump sums (period:amount, comma-separated)", value="")

    io_text = st.text_input(
        "Interest-only windows (start-end, comma-separated)", value="")

    st.header("Part-and-Part")
    pnp_amount = st.number_input(
        "IO tranche amount (€)", min_value=0.0, value=0.0, step=1000.0)
    pnp_window_text = st.text_input("IO tranche window (start-end)", value="")

    st.header("Cashback")
    cb_upfront_fixed = st.number_input(
        "Upfront cashback (€)", min_value=0.0, value=0.0, step=100.0)
    cb_upfront_pct = st.number_input(
        "Upfront cashback (% of drawdown)", min_value=0.0, value=0.0, step=0.1)
    cb_recurring_amt = st.number_input(
        "Recurring cashback per installment (€)", min_value=0.0, value=0.0, step=10.0)
    cb_recurring_n = st.number_input(
        "Recurring cashback for first N installments", min_value=0, value=0, step=1)
    cb_map_text = st.text_input(
        "Extra cashback at specific installments (period:amount)", value="")

# Parse complex inputs
lumps = {}
rates = {}
io_windows: List[Tuple[int, int]] = []
pnp_window = None
cb_map = {}

try:
    lumps = parse_period_amount_map(lump_text)
except Exception:
    st.warning("Check lump sums format (e.g., 24:5000, 60:10000)")

try:
    rates = parse_period_rate_map(rate_text)
except Exception:
    st.warning("Check variable rates format (e.g., 1:4.25, 25:5.1)")

try:
    io_windows = parse_ranges(io_text)
except Exception:
    st.warning("Check IO windows format (e.g., 1-12, 25-36)")

try:
    if pnp_window_text.strip():
        a, b = pnp_window_text.split("-")
        pnp_window = (int(a.strip()), int(b.strip()))
except Exception:
    st.warning("Check part-and-part window format (e.g., 1-60)")

try:
    cb_map = parse_period_amount_map(cb_map_text)
except Exception:
    st.warning("Check cashback map format (e.g., 1:2000, 12:100)")

# Build schedule
schedule = build_schedule(
    principal=principal,
    payments_per_year=payments_per_year,
    years=int(term_years),
    base_annual_rate_pct=rate,
    regular_overpay=regular_overpay,
    overpay_start_period=int(overpay_start),
    strategy=strategy,
    lump_sums=lumps,
    rate_changes=rates if rates else {1: rate},
    interest_only_windows=io_windows,
    part_and_part_amount=pnp_amount,
    part_and_part_window=pnp_window,
    drawdown_cashback_fixed=cb_upfront_fixed,
    drawdown_cashback_pct=cb_upfront_pct,
    recurring_cashback_first_n=int(cb_recurring_n),
    recurring_cashback_amount=cb_recurring_amt,
    cashback_period_amounts=cb_map,
    round_cents=True,
)

# Summary
T = totals(schedule)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Installments", T["num_installments"])
col2.metric("Total paid", f"€{T['total_payment']:,.2f}")
col3.metric("Cashback", f"€{T['total_cashback']:,.2f}")
col4.metric("Net paid", f"€{T['net_total_paid']:,.2f}")

col5, col6 = st.columns(2)
col5.metric("Total interest", f"€{T['total_interest']:,.2f}")
col6.metric("Principal repaid", f"€{T['total_principal']:,.2f}")

# Table
df = pd.DataFrame([
    {
        "Installment": r.number,
        "RateAnnualPct": r.rate_annual_pct,
        "BasePayment": r.base_payment,
        "Overpayment": r.overpay,
        "LumpSum": r.lump,
        "Interest": r.interest,
        "Principal": r.principal,
        "Cashback": r.cashback,
        "NetPaid": r.net_paid,
        "Balance": r.balance,
    }
    for r in schedule
])

st.subheader("Amortisation Schedule")
st.dataframe(df, use_container_width=True)

# Charts
st.subheader("Charts")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    fig1, ax1 = plt.subplots()
    dff = df[df["Installment"] >= 1].copy()
    dff["CumulativeNetPaid"] = dff["NetPaid"].cumsum()  # <-- NEW
    ax1.plot(dff["Installment"], dff["Balance"], label="Balance")
    ax1.plot(dff["Installment"], dff["CumulativeNetPaid"],
             label="Cumulative Net Paid")  # <-- NEW
    ax1.set_title("Remaining Balance")
    ax1.set_xlabel("Installment")
    ax1.set_ylabel("€")
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)
with plot_col2:
    fig2, ax2 = plt.subplots()
    dff = df[df["Installment"] >= 1]
    dff["TotalPayment"] = dff["Interest"] + dff["Principal"]
    ax2.plot(dff["Installment"], dff["Principal"], label="Principal")
    ax2.plot(dff["Installment"], dff["Interest"], label="Interest")
    ax2.plot(dff["Installment"], dff["TotalPayment"], label="Total Payment")
    ax2.set_title("Payment Breakdown per Installment")
    ax2.set_xlabel("Installment")
    ax2.set_ylabel("€ per installment")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# Downloads (Excel + CSV)
st.subheader("Export")

# Excel in-memory
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
    df2 = df.copy()
    df2["CumulativeInterest"] = df2["Interest"].cumsum()
    df2["CumulativeNetPaid"] = df2["NetPaid"].cumsum()
    df2.to_excel(writer, index=False, sheet_name="Schedule")
    pd.DataFrame({
        "NumInstallments": [T["num_installments"]],
        "TotalPayments": [T["total_payment"]],
        "TotalCashback": [T["total_cashback"]],
        "NetTotalPaid": [T["net_total_paid"]],
        "TotalInterest": [T["total_interest"]],
        "PrincipalRepaid": [T["total_principal"]],
    }).to_excel(writer, index=False, sheet_name="Summary")

st.download_button(
    label="Download Excel (.xlsx)",
    data=xlsx_buf.getvalue(),
    file_name="mortgage_schedule.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# CSV
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    label="Download CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="mortgage_schedule.csv",
    mime="text/csv",
)
