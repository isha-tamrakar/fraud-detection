import pandas as pd
import pytest
from analyze_fraud import score_transactions, summarize_results


def _scored(*rows):
    """Build a minimal scored DataFrame from (transaction_id, amount_usd, risk_label) tuples."""
    return pd.DataFrame(rows, columns=["transaction_id", "amount_usd", "risk_label"])


def _chargebacks(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


# ---------------------------------------------------------------------------
# summarize_results — transaction counts
# ---------------------------------------------------------------------------

def test_summarize_transaction_counts():
    scored = _scored(
        ("t1", 100.0, "low"),
        ("t2", 200.0, "low"),
        ("t3", 500.0, "medium"),
        ("t4", 300.0, "high"),
    )
    result = summarize_results(scored, _chargebacks())
    counts = dict(zip(result["risk_label"], result["transactions"]))
    assert counts["low"] == 2
    assert counts["medium"] == 1
    assert counts["high"] == 1


# ---------------------------------------------------------------------------
# summarize_results — amount totals and averages
# ---------------------------------------------------------------------------

def test_summarize_total_and_avg_amounts():
    scored = _scored(
        ("t1", 100.0, "low"),
        ("t2", 300.0, "low"),
        ("t3", 500.0, "high"),
    )
    result = summarize_results(scored, _chargebacks())
    low = result[result["risk_label"] == "low"].iloc[0]
    assert low["total_amount_usd"] == pytest.approx(400.0)
    assert low["avg_amount_usd"] == pytest.approx(200.0)

    high = result[result["risk_label"] == "high"].iloc[0]
    assert high["total_amount_usd"] == pytest.approx(500.0)
    assert high["avg_amount_usd"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# summarize_results — chargeback counts and rates
# ---------------------------------------------------------------------------

def test_summarize_chargeback_rate_zero():
    scored = _scored(("t1", 100.0, "low"))
    result = summarize_results(scored, _chargebacks())
    row = result[result["risk_label"] == "low"].iloc[0]
    assert row["chargebacks"] == 0
    assert row["chargeback_rate"] == pytest.approx(0.0)


def test_summarize_chargeback_rate_full():
    scored = _scored(("t1", 200.0, "high"))
    result = summarize_results(scored, _chargebacks("t1"))
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 1
    assert row["chargeback_rate"] == pytest.approx(1.0)


def test_summarize_chargeback_rate_partial():
    scored = _scored(
        ("t1", 100.0, "high"),
        ("t2", 200.0, "high"),
        ("t3", 300.0, "high"),
    )
    result = summarize_results(scored, _chargebacks("t1", "t3"))
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 2
    assert row["chargeback_rate"] == pytest.approx(2 / 3)


def test_summarize_chargebacks_only_counted_in_correct_bucket():
    scored = _scored(
        ("t1", 100.0, "low"),
        ("t2", 200.0, "medium"),
        ("t3", 300.0, "high"),
    )
    result = summarize_results(scored, _chargebacks("t3"))
    rates = dict(zip(result["risk_label"], result["chargeback_rate"]))
    assert rates["low"] == pytest.approx(0.0)
    assert rates["medium"] == pytest.approx(0.0)
    assert rates["high"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize_results — sort order
# ---------------------------------------------------------------------------

def test_summarize_sort_order_low_medium_high():
    scored = _scored(
        ("t1", 100.0, "high"),
        ("t2", 200.0, "low"),
        ("t3", 300.0, "medium"),
    )
    result = summarize_results(scored, _chargebacks())
    assert list(result["risk_label"]) == ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# score_transactions — integration
# ---------------------------------------------------------------------------

def test_score_transactions_adds_risk_columns():
    accounts = pd.DataFrame({"account_id": [1], "prior_chargebacks": [0]})
    transactions = pd.DataFrame({
        "transaction_id": ["t1"],
        "account_id": [1],
        "amount_usd": [100.0],
        "device_risk_score": [10],
        "is_international": [0],
        "velocity_24h": [1],
        "failed_logins_24h": [0],
    })
    result = score_transactions(transactions, accounts)
    assert "risk_score" in result.columns
    assert "risk_label" in result.columns


def test_score_transactions_clean_profile_is_low_risk():
    accounts = pd.DataFrame({"account_id": [1], "prior_chargebacks": [0]})
    transactions = pd.DataFrame({
        "transaction_id": ["t1"],
        "account_id": [1],
        "amount_usd": [50.0],
        "device_risk_score": [5],
        "is_international": [0],
        "velocity_24h": [1],
        "failed_logins_24h": [0],
    })
    result = score_transactions(transactions, accounts)
    assert result["risk_label"].iloc[0] == "low"


def test_score_transactions_high_risk_profile_is_high_risk():
    accounts = pd.DataFrame({"account_id": [1], "prior_chargebacks": [2]})
    transactions = pd.DataFrame({
        "transaction_id": ["t1"],
        "account_id": [1],
        "amount_usd": [1500.0],
        "device_risk_score": [85],
        "is_international": [1],
        "velocity_24h": [8],
        "failed_logins_24h": [6],
    })
    result = score_transactions(transactions, accounts)
    assert result["risk_label"].iloc[0] == "high"
    assert result["risk_score"].iloc[0] == 100
