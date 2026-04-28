import pandas as pd
from features import build_model_frame


def _frame(amount_usd, failed_logins_24h, prior_chargebacks=0):
    transactions = pd.DataFrame({
        "account_id": [1],
        "amount_usd": [amount_usd],
        "failed_logins_24h": [failed_logins_24h],
    })
    accounts = pd.DataFrame({
        "account_id": [1],
        "prior_chargebacks": [prior_chargebacks],
    })
    return build_model_frame(transactions, accounts)


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

def test_is_large_amount_below_threshold():
    assert _frame(999, 0)["is_large_amount"].iloc[0] == 0


def test_is_large_amount_at_threshold():
    assert _frame(1000, 0)["is_large_amount"].iloc[0] == 1


def test_is_large_amount_above_threshold():
    assert _frame(2500, 0)["is_large_amount"].iloc[0] == 1


# ---------------------------------------------------------------------------
# login_pressure
# ---------------------------------------------------------------------------

def test_login_pressure_none_at_zero():
    assert _frame(100, 0)["login_pressure"].iloc[0] == "none"


def test_login_pressure_low_at_one():
    assert _frame(100, 1)["login_pressure"].iloc[0] == "low"


def test_login_pressure_low_at_two():
    assert _frame(100, 2)["login_pressure"].iloc[0] == "low"


def test_login_pressure_high_above_two():
    assert _frame(100, 3)["login_pressure"].iloc[0] == "high"
    assert _frame(100, 10)["login_pressure"].iloc[0] == "high"


# ---------------------------------------------------------------------------
# merge behaviour
# ---------------------------------------------------------------------------

def test_prior_chargebacks_merged_from_accounts():
    result = _frame(100, 0, prior_chargebacks=3)
    assert result["prior_chargebacks"].iloc[0] == 3


def test_multiple_transactions_merged_correctly():
    transactions = pd.DataFrame({
        "account_id": [1, 2],
        "amount_usd": [500, 1500],
        "failed_logins_24h": [0, 3],
    })
    accounts = pd.DataFrame({
        "account_id": [1, 2],
        "prior_chargebacks": [0, 2],
    })
    result = build_model_frame(transactions, accounts)
    assert len(result) == 2
    assert result.loc[result["account_id"] == 2, "prior_chargebacks"].iloc[0] == 2
    assert result.loc[result["account_id"] == 2, "is_large_amount"].iloc[0] == 1
