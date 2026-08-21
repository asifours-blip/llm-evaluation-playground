from datetime import date, timedelta
from decimal import Decimal

import pytest

from rag_quality_lab.config.loaders import load_yaml_model
from rag_quality_lab.domain.models import (
    BudgetConfig,
    ModelPrice,
    PricingConfig,
    TokenUsage,
)
from rag_quality_lab.experiments.budget import (
    BudgetExceeded,
    BudgetLedger,
    PlannedCall,
    calculate_actual_cost,
    estimate_tokens_upper_bound,
    preflight_budget,
)


def pricing(
    *, input_rate: int = 3, output_rate: int = 6, verified_at: date | None = None
) -> PricingConfig:
    return PricingConfig(
        provider="test",
        currency="CNY",
        verified_at=verified_at or date.today(),
        source_url="https://example.com/pricing",
        models={
            "pro": ModelPrice(
                input_cache_hit=1,
                input_cache_miss=input_rate,
                output=output_rate,
            )
        },
    )


def test_preflight_uses_conservative_byte_token_estimate() -> None:
    estimate = estimate_tokens_upper_bound("中文")

    assert estimate == len("中文".encode())


def test_preflight_blocks_when_buffered_cost_exceeds_threshold() -> None:
    decision = preflight_budget(
        planned=[
            PlannedCall(
                model="pro",
                input_token_cap=3500,
                output_token_cap=256,
                count=192,
            )
        ],
        pricing=pricing(input_rate=3, output_rate=6),
        budget=BudgetConfig(hard_limit=Decimal("2.00")),
    )

    assert not decision.allowed
    assert decision.buffered_cost > Decimal("1.80")


def test_preflight_blocks_stale_pricing() -> None:
    decision = preflight_budget(
        planned=[PlannedCall(model="pro", input_token_cap=1, output_token_cap=1)],
        pricing=pricing(verified_at=date.today() - timedelta(days=8)),
        budget=BudgetConfig(hard_limit=20),
        on_date=date.today(),
    )

    assert not decision.allowed
    assert "stale" in decision.reason


def test_actual_cost_distinguishes_cache_hit_and_miss_tokens() -> None:
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        input_cache_hit_tokens=60,
        input_cache_miss_tokens=40,
    )

    cost = calculate_actual_cost(usage, pricing().models["pro"])

    assert cost == Decimal("0.000300")


def test_actual_cost_treats_unspecified_cache_breakdown_as_all_miss() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=20)

    assert calculate_actual_cost(usage, pricing().models["pro"]) == Decimal(
        "0.000420"
    )


def test_ledger_rejects_reservation_that_could_exceed_hard_limit() -> None:
    ledger = BudgetLedger(
        budget=BudgetConfig(hard_limit=Decimal("0.000010")),
        pricing=pricing(input_rate=3, output_rate=6),
    )

    with pytest.raises(BudgetExceeded):
        ledger.reserve(model="pro", input_token_cap=2, output_token_cap=1)


def test_ledger_reserves_and_settles_actual_usage() -> None:
    ledger = BudgetLedger(
        budget=BudgetConfig(hard_limit=Decimal("0.001")),
        pricing=pricing(input_rate=3, output_rate=6),
    )
    reservation = ledger.reserve(
        model="pro", input_token_cap=100, output_token_cap=20
    )

    actual = ledger.settle(
        reservation,
        model="pro",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=20,
            input_cache_hit_tokens=60,
            input_cache_miss_tokens=40,
        ),
    )

    assert reservation == Decimal("0.000420")
    assert actual == Decimal("0.000300")
    assert ledger.reserved == 0
    assert ledger.spent == actual


def test_ledger_atomically_reserves_and_settles_generation_and_judge() -> None:
    ledger = BudgetLedger(
        budget=BudgetConfig(hard_limit=Decimal("0.01")),
        pricing=pricing(input_rate=3, output_rate=6),
    )
    calls = [
        PlannedCall(model="pro", input_token_cap=100, output_token_cap=20),
        PlannedCall(model="pro", input_token_cap=80, output_token_cap=10),
    ]

    reservations = ledger.reserve_many(calls)
    actual = ledger.settle_many(
        reservations,
        [
            ("pro", TokenUsage(input_tokens=90, output_tokens=15)),
            ("pro", TokenUsage(input_tokens=70, output_tokens=8)),
        ],
    )

    assert reservations == [Decimal("0.000420"), Decimal("0.000300")]
    assert actual == Decimal("0.000618")
    assert ledger.reserved == 0
    assert ledger.spent == actual


def test_ledger_can_charge_full_reservation_when_usage_is_unavailable() -> None:
    ledger = BudgetLedger(
        budget=BudgetConfig(hard_limit=Decimal("0.01")),
        pricing=pricing(input_rate=3, output_rate=6),
    )
    reservations = ledger.reserve_many(
        [PlannedCall(model="pro", input_token_cap=100, output_token_cap=20)]
    )

    charged = ledger.charge_reserved(reservations)

    assert charged == Decimal("0.000420")
    assert ledger.reserved == 0
    assert ledger.spent == charged


def test_ledger_records_and_raises_when_actual_usage_exceeds_cap() -> None:
    ledger = BudgetLedger(
        budget=BudgetConfig(hard_limit=Decimal("0.000010")),
        pricing=pricing(input_rate=3, output_rate=6),
    )
    reservation = ledger.reserve(model="pro", input_token_cap=1, output_token_cap=1)

    with pytest.raises(BudgetExceeded, match="actual"):
        ledger.settle(
            reservation,
            model="pro",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )

    assert ledger.spent == Decimal("0.000420")


def test_budget_rejects_currency_mismatch_and_unknown_model() -> None:
    with pytest.raises(ValueError, match="currency"):
        BudgetLedger(
            budget=BudgetConfig(currency="USD", hard_limit=20),
            pricing=pricing(),
        )

    with pytest.raises(ValueError, match="missing price"):
        preflight_budget(
            planned=[
                PlannedCall(model="unknown", input_token_cap=1, output_token_cap=1)
            ],
            pricing=pricing(),
            budget=BudgetConfig(hard_limit=20),
        )


def test_official_peak_pricing_matches_full_plan_estimate() -> None:
    official = load_yaml_model(
        "configs/pricing/deepseek-2026-08-21.yaml", PricingConfig
    )
    planned = [
        PlannedCall(
            model="deepseek-v4-flash",
            input_token_cap=2500,
            output_token_cap=512,
            count=132,
        ),
        PlannedCall(
            model="deepseek-v4-pro",
            input_token_cap=2500,
            output_token_cap=512,
            count=96,
        ),
        PlannedCall(
            model="deepseek-v4-pro",
            input_token_cap=3500,
            output_token_cap=256,
            count=192,
        ),
    ]

    decision = preflight_budget(
        planned=planned,
        pricing=official,
        budget=BudgetConfig(hard_limit=20),
        on_date=date(2026, 8, 21),
    )

    assert official.rate_basis == "peak"
    assert decision.unbuffered_cost == Decimal("12.460464")
    assert decision.buffered_cost == Decimal("15.57558000")
    assert decision.allowed
