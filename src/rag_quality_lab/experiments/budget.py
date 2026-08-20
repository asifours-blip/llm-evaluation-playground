"""Conservative preflight and actual-cost budget accounting."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from rag_quality_lab.domain.models import (
    BudgetConfig,
    ModelPrice,
    PricingConfig,
    TokenUsage,
)

TOKENS_PER_MILLION = Decimal("1000000")


class PlannedCall(BaseModel):
    """Capped token usage for one repeated provider call shape."""

    model: str
    input_token_cap: int = Field(gt=0)
    output_token_cap: int = Field(ge=0)
    count: int = Field(default=1, gt=0)


class PreflightDecision(BaseModel):
    """Auditable decision made before any paid call is scheduled."""

    allowed: bool
    unbuffered_cost: Decimal
    buffered_cost: Decimal
    threshold: Decimal
    hard_limit: Decimal
    reason: str


class BudgetExceeded(RuntimeError):
    """Raised before scheduling a call whose cap could exceed the budget."""


def estimate_tokens_upper_bound(text: str) -> int:
    """Use UTF-8 bytes as a conservative, tokenizer-independent upper bound."""

    return len(text.encode("utf-8"))


def planned_call_cost(call: PlannedCall, pricing: PricingConfig) -> Decimal:
    """Calculate worst-case cost using cache-miss input pricing."""

    price = _model_price(pricing, call.model)
    one_call = (
        Decimal(call.input_token_cap) * price.input_cache_miss
        + Decimal(call.output_token_cap) * price.output
    ) / TOKENS_PER_MILLION
    return one_call * call.count


def preflight_budget(
    *,
    planned: list[PlannedCall],
    pricing: PricingConfig,
    budget: BudgetConfig,
    on_date: date | None = None,
    max_pricing_age_days: int = 7,
) -> PreflightDecision:
    """Decide whether the complete buffered plan fits the configured threshold."""

    if pricing.currency != budget.currency:
        raise ValueError("budget and pricing currency must match")
    unbuffered = sum(
        (planned_call_cost(call, pricing) for call in planned), start=Decimal("0")
    )
    buffered = unbuffered * budget.safety_multiplier
    threshold = budget.hard_limit * budget.preflight_fraction
    evaluation_date = on_date or date.today()
    if pricing.is_stale(evaluation_date, max_age_days=max_pricing_age_days):
        return PreflightDecision(
            allowed=False,
            unbuffered_cost=unbuffered,
            buffered_cost=buffered,
            threshold=threshold,
            hard_limit=budget.hard_limit,
            reason="pricing is stale",
        )
    allowed = buffered <= threshold
    return PreflightDecision(
        allowed=allowed,
        unbuffered_cost=unbuffered,
        buffered_cost=buffered,
        threshold=threshold,
        hard_limit=budget.hard_limit,
        reason="allowed" if allowed else "buffered cost exceeds preflight threshold",
    )


def calculate_actual_cost(usage: TokenUsage, price: ModelPrice) -> Decimal:
    """Calculate cost from provider-reported cache and output token usage."""

    hit_tokens = usage.input_cache_hit_tokens
    miss_tokens = usage.input_cache_miss_tokens
    if hit_tokens == 0 and miss_tokens == 0:
        miss_tokens = usage.input_tokens
    hit_rate = price.input_cache_hit or price.input_cache_miss
    return (
        Decimal(hit_tokens) * hit_rate
        + Decimal(miss_tokens) * price.input_cache_miss
        + Decimal(usage.output_tokens) * price.output
    ) / TOKENS_PER_MILLION


class BudgetLedger:
    """Track actual spend and outstanding capped-call reservations."""

    def __init__(self, *, budget: BudgetConfig, pricing: PricingConfig) -> None:
        if budget.currency != pricing.currency:
            raise ValueError("budget and pricing currency must match")
        self.budget = budget
        self.pricing = pricing
        self.spent = Decimal("0")
        self.reserved = Decimal("0")

    def reserve(self, *, model: str, input_token_cap: int, output_token_cap: int) -> Decimal:
        reservation = planned_call_cost(
            PlannedCall(
                model=model,
                input_token_cap=input_token_cap,
                output_token_cap=output_token_cap,
            ),
            self.pricing,
        )
        if self.spent + self.reserved + reservation > self.budget.hard_limit:
            raise BudgetExceeded("next capped call could exceed the hard budget")
        self.reserved += reservation
        return reservation

    def settle(
        self,
        reservation: Decimal,
        *,
        model: str,
        usage: TokenUsage,
    ) -> Decimal:
        if reservation < 0 or reservation > self.reserved:
            raise ValueError("reservation is not outstanding")
        actual = calculate_actual_cost(usage, _model_price(self.pricing, model))
        self.reserved -= reservation
        self.spent += actual
        if self.spent > self.budget.hard_limit:
            raise BudgetExceeded("actual cost exceeded the hard budget")
        return actual


def _model_price(pricing: PricingConfig, model: str) -> ModelPrice:
    try:
        return pricing.models[model]
    except KeyError as error:
        raise ValueError(f"missing price for model: {model}") from error
