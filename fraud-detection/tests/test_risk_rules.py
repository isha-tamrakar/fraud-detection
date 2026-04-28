import pytest
from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_low():
    assert label_risk(0) == "low"
    assert label_risk(10) == "low"
    assert label_risk(29) == "low"


def test_label_risk_medium():
    assert label_risk(30) == "medium"
    assert label_risk(35) == "medium"
    assert label_risk(59) == "medium"


def test_label_risk_high():
    assert label_risk(60) == "high"
    assert label_risk(75) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — base case
# ---------------------------------------------------------------------------

def test_base_transaction_scores_zero():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# score_transaction — device_risk_score
# ---------------------------------------------------------------------------

def test_low_risk_device_adds_nothing():
    assert score_transaction(_base_tx(device_risk_score=39)) == 0


def test_medium_risk_device_adds_10():
    assert score_transaction(_base_tx(device_risk_score=40)) == 10
    assert score_transaction(_base_tx(device_risk_score=55)) == 10
    assert score_transaction(_base_tx(device_risk_score=69)) == 10


def test_high_risk_device_adds_25():
    assert score_transaction(_base_tx(device_risk_score=70)) == 25
    assert score_transaction(_base_tx(device_risk_score=85)) == 25


def test_high_risk_device_adds_risk():
    assert score_transaction(_base_tx(device_risk_score=85)) > score_transaction(_base_tx(device_risk_score=10))


# ---------------------------------------------------------------------------
# score_transaction — is_international
# ---------------------------------------------------------------------------

def test_domestic_transaction_adds_nothing():
    assert score_transaction(_base_tx(is_international=0)) == 0


def test_international_transaction_adds_15():
    assert score_transaction(_base_tx(is_international=1)) == 15


def test_international_adds_risk():
    assert score_transaction(_base_tx(is_international=1)) > score_transaction(_base_tx(is_international=0))


# ---------------------------------------------------------------------------
# score_transaction — amount_usd
# ---------------------------------------------------------------------------

def test_small_amount_adds_nothing():
    assert score_transaction(_base_tx(amount_usd=499)) == 0


def test_medium_amount_adds_10():
    assert score_transaction(_base_tx(amount_usd=500)) == 10
    assert score_transaction(_base_tx(amount_usd=750)) == 10
    assert score_transaction(_base_tx(amount_usd=999)) == 10


def test_large_amount_adds_25():
    assert score_transaction(_base_tx(amount_usd=1000)) == 25
    assert score_transaction(_base_tx(amount_usd=1200)) == 25


# ---------------------------------------------------------------------------
# score_transaction — velocity_24h
# ---------------------------------------------------------------------------

def test_low_velocity_adds_nothing():
    assert score_transaction(_base_tx(velocity_24h=2)) == 0


def test_medium_velocity_adds_5():
    assert score_transaction(_base_tx(velocity_24h=3)) == 5
    assert score_transaction(_base_tx(velocity_24h=5)) == 5


def test_high_velocity_adds_20():
    assert score_transaction(_base_tx(velocity_24h=6)) == 20
    assert score_transaction(_base_tx(velocity_24h=8)) == 20


def test_high_velocity_adds_risk():
    assert score_transaction(_base_tx(velocity_24h=8)) > score_transaction(_base_tx(velocity_24h=1))


# ---------------------------------------------------------------------------
# score_transaction — failed_logins_24h
# ---------------------------------------------------------------------------

def test_no_failed_logins_adds_nothing():
    assert score_transaction(_base_tx(failed_logins_24h=0)) == 0
    assert score_transaction(_base_tx(failed_logins_24h=1)) == 0


def test_moderate_failed_logins_adds_10():
    assert score_transaction(_base_tx(failed_logins_24h=2)) == 10
    assert score_transaction(_base_tx(failed_logins_24h=4)) == 10


def test_high_failed_logins_adds_20():
    assert score_transaction(_base_tx(failed_logins_24h=5)) == 20
    assert score_transaction(_base_tx(failed_logins_24h=9)) == 20


# ---------------------------------------------------------------------------
# score_transaction — prior_chargebacks
# ---------------------------------------------------------------------------

def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(_base_tx(prior_chargebacks=0)) == 0


def test_one_prior_chargeback_adds_10():
    assert score_transaction(_base_tx(prior_chargebacks=1)) == 10


def test_multiple_prior_chargebacks_adds_20():
    assert score_transaction(_base_tx(prior_chargebacks=2)) == 20
    assert score_transaction(_base_tx(prior_chargebacks=5)) == 20


def test_prior_chargebacks_add_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one_cb = score_transaction(_base_tx(prior_chargebacks=1))
    two_cb = score_transaction(_base_tx(prior_chargebacks=2))
    assert one_cb > clean
    assert two_cb > one_cb


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_clamped_at_100():
    # device(+25) + intl(+15) + amount(+25) + velocity(+20) + logins(+20) + chargebacks(+20) = 125
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


def test_score_never_negative():
    assert score_transaction(_base_tx()) >= 0
