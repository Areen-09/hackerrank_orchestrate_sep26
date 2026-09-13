"""
Deterministic 90-Day Financial Decision Engine for HackerRank Orchestrate.
Implements exact simulation, cash flow forecasting, payment option evaluation,
spending changes search, and ranking rules from problem_statement.md §176-§216.
"""

import os
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def resolve_path(p: str) -> str:
    """Finds path whether running from repo root, inside code/ directory, or elsewhere."""
    if os.path.exists(p):
        return p
    parent_p = os.path.join("..", p)
    if os.path.exists(parent_p):
        return parent_p
    script_rel = os.path.join(os.path.dirname(__file__), "..", p)
    if os.path.exists(script_rel):
        return script_rel
    return p


class ExchangeRateManager:
    def __init__(self, rates_csv_path: str = "dataset/exchange_rates.csv"):
        rates_csv_path = resolve_path(rates_csv_path)
        self.rates = {}
        if os.path.exists(rates_csv_path):
            df = pd.read_csv(rates_csv_path)
            for _, r in df.iterrows():
                key = (str(r["rate_date"]), str(r["from_currency"]).upper(), str(r["to_currency"]).upper())
                self.rates[key] = float(r["rate"])

    def convert(self, amount: float, from_curr: str, to_curr: str, date_str: str) -> float:
        from_curr = from_curr.upper()
        to_curr = to_curr.upper()
        if from_curr == to_curr or amount == 0:
            return amount
        key = (date_str, from_curr, to_curr)
        if key in self.rates:
            return amount * self.rates[key]
        # Reverse key
        rev_key = (date_str, to_curr, from_curr)
        if rev_key in self.rates and self.rates[rev_key] > 0:
            return amount / self.rates[rev_key]
        # Fallback to any available date for this pair
        for (d, fc, tc), r in self.rates.items():
            if fc == from_curr and tc == to_curr:
                return amount * r
        return amount


def format_payment_amount(amount: float) -> str:
    """Formats amount as integer if whole, or 2 decimal places if fractional."""
    rounded = round(amount, 2)
    if abs(rounded - round(rounded)) < 1e-5:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


class FinancialEngine:
    def __init__(
        self,
        profiles_path: str = "dataset/financial_profiles.csv",
        events_path: str = "dataset/financial_events.csv",
        options_path: str = "dataset/request_payment_options.csv",
        rates_path: str = "dataset/exchange_rates.csv",
        ocr_overrides: Optional[Dict[str, float]] = None,
        message_insights: Optional[Dict[str, Any]] = None
    ):
        self.profiles = pd.read_csv(resolve_path(profiles_path)).set_index("user_id")
        self.events = pd.read_csv(resolve_path(events_path))
        self.options = pd.read_csv(resolve_path(options_path))
        self.fx = ExchangeRateManager(rates_path)
        self.ocr_overrides = ocr_overrides or {}
        self.message_insights = message_insights or {}

        # Apply OCR overrides to missing event amounts
        for ev_id, amt in self.ocr_overrides.items():
            mask = self.events["event_id"] == ev_id
            if mask.any():
                self.events.loc[mask, "amount"] = amt

    def get_user_cashflow_schedule(
        self,
        user_id: str,
        request_date_str: str,
        forecast_days: int = 90,
        spending_changes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        prof = self.profiles.loc[user_id]
        home_curr = str(prof["home_currency"])
        req_dt = datetime.strptime(request_date_str, "%Y-%m-%d")
        end_dt = req_dt + timedelta(days=forecast_days)

        # Initialize daily net flow
        daily_flow: Dict[str, float] = {}
        curr_dt = req_dt
        while curr_dt <= end_dt:
            daily_flow[curr_dt.strftime("%Y-%m-%d")] = 0.0
            curr_dt += timedelta(days=1)

        # Protected categories + willing to stop/reduce categories
        prot_cats = set(c.strip() for c in str(prof["expense_categories_to_protect"]).split("|") if c.strip() and c.strip() != "nan")
        willing_stop = set(c.strip() for c in str(prof["expense_categories_user_is_willing_to_stop"]).split("|") if c.strip() and c.strip() != "nan")
        willing_reduce = set(c.strip() for c in str(prof["expense_categories_user_is_willing_to_reduce"]).split("|") if c.strip() and c.strip() != "nan")
        relevant_cats = prot_cats.union(willing_stop).union(willing_reduce)

        # Parse active spending changes
        stopped_events = set()
        reduced_events: Dict[str, float] = {}
        if spending_changes:
            for sc in spending_changes:
                if sc.startswith("stop:"):
                    stopped_events.add(sc.split(":")[1])
                elif sc.startswith("reduce_to:"):
                    parts = sc.split(":")
                    reduced_events[parts[1]] = float(parts[2])

        u_events = self.events[self.events["user_id"] == user_id].copy()

        # 1. Pending debits already present in events
        pending_debits = u_events[
            (u_events["status"] == "pending") &
            (u_events["direction"] == "debit")
        ]
        for _, pev in pending_debits.iterrows():
            amt = float(pev["amount"]) if pd.notna(pev["amount"]) else 0.0
            s_date = str(pev["settlement_date"]) if pd.notna(pev["settlement_date"]) else request_date_str
            effective_date = max(s_date, request_date_str)
            if effective_date in daily_flow:
                amt_home = self.fx.convert(amt, str(pev["currency"]), home_curr, effective_date)
                daily_flow[effective_date] -= amt_home

        # 2. Confirmed future events on or after request_date
        future_events = u_events[
            (u_events["settlement_date"] >= request_date_str) &
            (u_events["status"].isin(["settled", "scheduled"]))
        ]
        future_salary_dates = set()
        for _, fev in future_events.iterrows():
            f_id = str(fev["event_id"])
            if f_id in stopped_events:
                continue
            amt = float(fev["amount"]) if pd.notna(fev["amount"]) else 0.0
            if f_id in reduced_events:
                amt = reduced_events[f_id]

            s_date = str(fev["settlement_date"])
            if s_date in daily_flow:
                amt_home = self.fx.convert(amt, str(fev["currency"]), home_curr, s_date)
                if fev["direction"] == "credit" and fev["category"] == "salary":
                    daily_flow[s_date] += amt_home
                    future_salary_dates.add(s_date)
                elif fev["direction"] == "debit":
                    daily_flow[s_date] -= amt_home

        # 3. Detect confirmed monthly salary and project across all months
        all_salaries = u_events[
            (u_events["category"] == "salary") &
            (~u_events["status"].isin(["cancelled", "failed", "unrealized"]))
        ].sort_values("settlement_date")

        if len(all_salaries) > 0:
            last_sal = all_salaries.iloc[-1]
            last_sal_amt = float(last_sal["amount"])
            sal_curr = str(last_sal["currency"])
            # Check message adjustments for salary
            for m_id, m_data in self.message_insights.items():
                if m_data.get("user_id") == user_id and m_data.get("salary_adjustment_amount"):
                    last_sal_amt = float(m_data["salary_adjustment_amount"])

            sal_day = datetime.strptime(str(last_sal["settlement_date"]), "%Y-%m-%d").day

            # Project salary on sal_day for every month in forecast period
            curr_dt = req_dt
            while curr_dt <= end_dt:
                if curr_dt.day == sal_day and curr_dt >= req_dt:
                    d_str = curr_dt.strftime("%Y-%m-%d")
                    # If this salary date was not already included from future_events
                    if d_str in daily_flow and d_str not in future_salary_dates:
                        amt_home = self.fx.convert(last_sal_amt, sal_curr, home_curr, d_str)
                        daily_flow[d_str] += amt_home
                curr_dt += timedelta(days=1)

        # 4. Recurring debits projection from historical records for relevant categories
        hist_events = u_events[
            (u_events["settlement_date"] < request_date_str) &
            (~u_events["status"].isin(["cancelled", "failed", "unrealized"]))
        ]

        for cat, grp in hist_events[hist_events["direction"] == "debit"].groupby("category"):
            if cat not in relevant_cats:
                continue
            grp = grp.sort_values("settlement_date")
            last_ev = grp.iloc[-1]
            last_ev_id = str(last_ev["event_id"])
            if last_ev_id in stopped_events:
                continue

            last_dt = datetime.strptime(str(last_ev["settlement_date"]), "%Y-%m-%d")

            if cat in ["groceries", "transport"]:
                base_amt = float(grp["amount"].mean())
            else:
                base_amt = float(last_ev["amount"])

            if last_ev_id in reduced_events:
                base_amt = reduced_events[last_ev_id]

            amt_home = self.fx.convert(base_amt, str(last_ev["currency"]), home_curr, request_date_str)

            if cat in ["groceries", "transport"]:
                next_dt = last_dt + timedelta(days=7)
                while next_dt <= end_dt:
                    if next_dt >= req_dt:
                        d_str = next_dt.strftime("%Y-%m-%d")
                        if d_str in daily_flow:
                            daily_flow[d_str] -= amt_home
                    next_dt += timedelta(days=7)
            elif cat in ["dining"]:
                next_dt = last_dt + timedelta(days=14)
                while next_dt <= end_dt:
                    if next_dt >= req_dt:
                        d_str = next_dt.strftime("%Y-%m-%d")
                        if d_str in daily_flow:
                            daily_flow[d_str] -= amt_home
                    next_dt += timedelta(days=14)
            else:
                day_m = last_dt.day
                curr_dt = req_dt
                while curr_dt <= end_dt:
                    if curr_dt.day == day_m and curr_dt >= req_dt:
                        d_str = curr_dt.strftime("%Y-%m-%d")
                        if d_str in daily_flow:
                            daily_flow[d_str] -= amt_home
                    curr_dt += timedelta(days=1)

        return daily_flow

    def simulate_trajectory(
        self,
        starting_balance: float,
        daily_flow: Dict[str, float],
        payments: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        balances: Dict[str, float] = {}
        current_bal = starting_balance
        payments = payments or {}

        for d_str in sorted(daily_flow.keys()):
            current_bal += daily_flow[d_str]
            if d_str in payments:
                current_bal -= payments[d_str]
            balances[d_str] = current_bal

        return balances

    def calculate_amount_safe_to_pay(
        self,
        starting_balance: float,
        min_balance: float,
        daily_flow: Dict[str, float],
        requested_amount: float
    ) -> float:
        balances = self.simulate_trajectory(starting_balance, daily_flow)
        min_headroom = min(b - min_balance for b in balances.values())
        safe = max(0.0, min(requested_amount, min_headroom))
        return round(safe, 2)

    def find_earliest_full_payment_date(
        self,
        starting_balance: float,
        min_balance: float,
        daily_flow: Dict[str, float],
        requested_amount: float
    ) -> Optional[str]:
        dates = sorted(daily_flow.keys())
        for d_str in dates:
            test_payments = {d_str: requested_amount}
            balances = self.simulate_trajectory(starting_balance, daily_flow, payments=test_payments)
            safe = all(balances[d] >= min_balance for d in dates if d >= d_str)
            if safe:
                return d_str
        return None

    def find_candidate_spending_changes(self, user_id: str, request_date_str: str) -> List[List[str]]:
        prof = self.profiles.loc[user_id]
        willing_stop = [c.strip() for c in str(prof["expense_categories_user_is_willing_to_stop"]).split("|") if c.strip() and c.strip() != "nan"]
        willing_reduce = [c.strip() for c in str(prof["expense_categories_user_is_willing_to_reduce"]).split("|") if c.strip() and c.strip() != "nan"]

        u_events = self.events[
            (self.events["user_id"] == user_id) &
            (self.events["settlement_date"] <= request_date_str) &
            (~self.events["status"].isin(["cancelled", "failed", "unrealized"]))
        ].sort_values("settlement_date")

        possible_actions = []
        for cat in set(willing_stop + willing_reduce):
            cat_events = u_events[u_events["category"] == cat]
            if len(cat_events) == 0:
                continue
            last_ev = cat_events.iloc[-1]
            ev_id = str(last_ev["event_id"])
            flex = str(last_ev["flexibility"])

            if cat in willing_stop and flex in ["stoppable", "reducible_or_stoppable"]:
                possible_actions.append(f"stop:{ev_id}")

            if cat in willing_reduce and flex in ["reducible", "reducible_or_stoppable"]:
                min_amt = last_ev["minimum_allowed_amount"]
                if pd.notna(min_amt):
                    formatted_min = format_payment_amount(float(min_amt))
                    possible_actions.append(f"reduce_to:{ev_id}:{formatted_min}")

        # Up to 3 actions, mutually exclusive per event
        combos: List[List[str]] = [[]]
        for a in possible_actions:
            combos.append([a])

        for i in range(len(possible_actions)):
            for j in range(i + 1, len(possible_actions)):
                a1, a2 = possible_actions[i], possible_actions[j]
                if a1.split(":")[1] != a2.split(":")[1]:
                    combos.append([a1, a2])

        return combos

    def evaluate_request(self, request_row: pd.Series) -> Dict[str, Any]:
        req_id = str(request_row["request_id"])
        user_id = str(request_row["user_id"])
        req_date = str(request_row["request_date"])
        req_amt = float(request_row["requested_amount"])
        desired_date = str(request_row["desired_completion_date"])
        allows_partial = bool(request_row["allows_partial_payment"])

        prof = self.profiles.loc[user_id]
        home_curr = str(prof["home_currency"])
        avail_bal = float(prof["current_available_balance"])
        min_bal = float(prof["minimum_balance_to_keep"])
        user_methods = [m.strip() for m in str(prof["payment_methods_user_will_consider"]).split("|")]
        max_inst_months = float(prof["max_installment_months"]) if pd.notna(prof["max_installment_months"]) and str(prof["max_installment_months"]).strip() != "" else 0

        # Baseline cash flow before optional spending changes
        baseline_flow = self.get_user_cashflow_schedule(user_id, req_date, spending_changes=None)
        amount_safe = self.calculate_amount_safe_to_pay(avail_bal, min_bal, baseline_flow, req_amt)
        earliest_full_date = self.find_earliest_full_payment_date(avail_bal, min_bal, baseline_flow, req_amt)

        candidate_plans: List[Dict[str, Any]] = []

        # Candidate 1: Immediate Full Payment without spending changes
        if "full_payment" in user_methods and amount_safe >= req_amt:
            p_str = f"{req_date}:{format_payment_amount(req_amt)}"
            candidate_plans.append({
                "affordability_status": "affordable_now",
                "recommended_payment_method": "full_payment",
                "payment_plan": p_str,
                "earliest_date_for_full_payment": req_date,
                "spending_changes_needed": "none",
                "completion_date": req_date,
                "requires_spending_changes": False,
                "total_cost": req_amt,
                "start_date": req_date,
                "num_payments": 1,
                "option_id_rank": 999999
            })

        # Candidate 2: Partial Payment without spending changes
        if (
            allows_partial
            and "partial_payment" in user_methods
            and 0 < amount_safe < req_amt
            and earliest_full_date
            and earliest_full_date <= desired_date
        ):
            rem_amt = req_amt - amount_safe
            p_str = f"{req_date}:{format_payment_amount(amount_safe)}|{earliest_full_date}:{format_payment_amount(rem_amt)}"
            candidate_plans.append({
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "partial_payment",
                "payment_plan": p_str,
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": "none",
                "completion_date": earliest_full_date,
                "requires_spending_changes": False,
                "total_cost": req_amt,
                "start_date": req_date,
                "num_payments": 2,
                "option_id_rank": 999998
            })

        # Candidate 3: Installment options from request_payment_options.csv (without spending changes)
        if "installments" in user_methods and max_inst_months > 0:
            req_options = self.options[self.options["request_id"] == req_id]
            for _, opt in req_options.iterrows():
                opt_id = str(opt["payment_option_id"])
                opt_method = str(opt["payment_method"])
                if opt_method != "installments":
                    continue
                num_payments = int(opt["number_of_payments"])
                if num_payments > max_inst_months:
                    continue

                inst_amt = float(opt["payment_amount"])
                first_date_str = str(opt["first_payment_date"])
                freq_days = float(opt["payment_frequency_days"])
                total_payable = float(opt["total_payable_amount"])

                first_dt = datetime.strptime(first_date_str, "%Y-%m-%d")
                inst_schedule: Dict[str, float] = {}
                plan_parts: List[str] = []
                for p_idx in range(num_payments):
                    p_dt = first_dt + timedelta(days=int(p_idx * freq_days))
                    p_date_str = p_dt.strftime("%Y-%m-%d")
                    inst_schedule[p_date_str] = inst_schedule.get(p_date_str, 0.0) + inst_amt
                    plan_parts.append(f"{p_date_str}:{format_payment_amount(inst_amt)}")

                traj = self.simulate_trajectory(avail_bal, baseline_flow, payments=inst_schedule)
                is_inst_safe = all(bal >= min_bal for bal in traj.values())

                if is_inst_safe:
                    last_pay_date = plan_parts[-1].split(":")[0]
                    opt_num = int(opt_id.replace("payment_option_", "")) if "payment_option_" in opt_id else 9999

                    candidate_plans.append({
                        "affordability_status": "affordable_with_plan",
                        "recommended_payment_method": "installments",
                        "payment_plan": "|".join(plan_parts),
                        "earliest_date_for_full_payment": earliest_full_date or "",
                        "spending_changes_needed": "none",
                        "completion_date": last_pay_date,
                        "requires_spending_changes": False,
                        "total_cost": total_payable,
                        "start_date": first_date_str,
                        "num_payments": num_payments,
                        "option_id_rank": opt_num
                    })

        # Candidate 4: Spending changes (if immediate full payment was unsafe)
        if amount_safe < req_amt:
            spending_combos = self.find_candidate_spending_changes(user_id, req_date)
            for combo in spending_combos:
                if not combo:
                    continue
                sc_flow = self.get_user_cashflow_schedule(user_id, req_date, spending_changes=combo)
                sc_amount_safe = self.calculate_amount_safe_to_pay(avail_bal, min_bal, sc_flow, req_amt)
                sc_str = "|".join(combo)

                if "full_payment" in user_methods and sc_amount_safe >= req_amt:
                    candidate_plans.append({
                        "affordability_status": "affordable_with_plan",
                        "recommended_payment_method": "full_payment",
                        "payment_plan": f"{req_date}:{format_payment_amount(req_amt)}",
                        "earliest_date_for_full_payment": earliest_full_date or req_date,
                        "spending_changes_needed": sc_str,
                        "completion_date": req_date,
                        "requires_spending_changes": True,
                        "total_cost": req_amt,
                        "start_date": req_date,
                        "num_payments": 1,
                        "option_id_rank": 999990
                    })

        # Candidate 5: Wait until earliest_date_for_full_payment (only if safe LATER, strictly after request_date, and on or before desired_date)
        if (
            "full_payment" in user_methods
            and earliest_full_date
            and earliest_full_date > req_date
            and earliest_full_date <= desired_date
        ):
            candidate_plans.append({
                "affordability_status": "affordable_later",
                "recommended_payment_method": "wait",
                "payment_plan": f"{earliest_full_date}:{format_payment_amount(req_amt)}",
                "earliest_date_for_full_payment": earliest_full_date,
                "spending_changes_needed": "none",
                "completion_date": earliest_full_date,
                "requires_spending_changes": False,
                "total_cost": req_amt,
                "start_date": earliest_full_date,
                "num_payments": 1,
                "option_id_rank": 999997
            })


        def ranking_key(plan: Dict[str, Any]):
            # 1. Complete by desired_completion_date
            completes_by_deadline = 0 if plan["completion_date"] <= desired_date else 1
            # 2. Require no spending changes
            no_spending_changes = 0 if not plan["requires_spending_changes"] else 1
            # 3. Minimize total amount paid
            cost = plan["total_cost"]
            # 4. Start earlier
            start_date = plan["start_date"]
            # 5. Fewer payments
            num_pay = plan["num_payments"]
            # 6. Lowest payment_option_id
            opt_rank = plan["option_id_rank"]
            return (completes_by_deadline, no_spending_changes, cost, start_date, num_pay, opt_rank)

        if candidate_plans:
            candidate_plans.sort(key=ranking_key)
            best = candidate_plans[0]
            earliest_out = req_date if best["affordability_status"] == "affordable_now" else (best["earliest_date_for_full_payment"] or "")

            return {
                "request_id": req_id,
                "amount_safe_to_pay": amount_safe,
                "affordability_status": best["affordability_status"],
                "recommended_payment_method": best["recommended_payment_method"],
                "payment_plan": best["payment_plan"],
                "earliest_date_for_full_payment": earliest_out,
                "spending_changes_needed": best["spending_changes_needed"],
                "user_id": user_id,
                "home_currency": home_curr,
                "minimum_balance_to_keep": min_bal,
                "requested_amount": req_amt,
                "desired_completion_date": desired_date
            }

        # Fallback: Not Affordable
        return {
            "request_id": req_id,
            "amount_safe_to_pay": amount_safe,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "user_id": user_id,
            "home_currency": home_curr,
            "minimum_balance_to_keep": min_bal,
            "requested_amount": req_amt,
            "desired_completion_date": desired_date
        }
