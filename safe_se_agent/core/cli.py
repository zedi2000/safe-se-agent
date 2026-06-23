from __future__ import annotations

import argparse

from safe_se_agent.core.memos_memory import MEMORY_BACKEND_CHOICES
from safe_se_agent.core.promotion import PromotionPolicyConfig


def add_memory_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-backend", choices=list(MEMORY_BACKEND_CHOICES), default="simple")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--retrieval-search-type",
        choices=["similarity", "similarity_score_threshold", "mmr"],
        default="similarity_score_threshold",
    )
    parser.add_argument("--retrieval-score-threshold", type=float, default=0.35)


def add_promotion_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--promotion-policy", choices=["none", "guard"], default="none")
    parser.add_argument("--promotion-min-support", type=int, default=2)
    parser.add_argument("--promotion-min-source-diversity", type=int, default=2)
    parser.add_argument("--promotion-similarity-threshold", type=float, default=0.38)
    parser.add_argument("--promotion-min-scope-confidence", type=float, default=0.55)
    parser.add_argument("--promotion-max-conflict-score", type=float, default=0.65)
    parser.add_argument("--promotion-max-safety-impact-score", type=float, default=0.80)
    parser.add_argument("--promotion-max-refresh-risk", type=float, default=0.70)


def promotion_policy_config_from_args(args: argparse.Namespace) -> PromotionPolicyConfig | None:
    if getattr(args, "promotion_policy", "none") != "guard":
        return None
    return PromotionPolicyConfig(
        min_support_count=args.promotion_min_support,
        min_source_diversity=args.promotion_min_source_diversity,
        similarity_threshold=args.promotion_similarity_threshold,
        min_scope_confidence=args.promotion_min_scope_confidence,
        max_conflict_score=args.promotion_max_conflict_score,
        max_safety_impact_score=args.promotion_max_safety_impact_score,
        max_refresh_risk=args.promotion_max_refresh_risk,
    )


def promotion_policy_config_for_summary(args: argparse.Namespace) -> dict[str, object]:
    config = promotion_policy_config_from_args(args)
    if config is None:
        return {"promotion_policy": "none"}
    return {
        "promotion_policy": "guard",
        "promotion_min_support": config.min_support_count,
        "promotion_min_source_diversity": config.min_source_diversity,
        "promotion_similarity_threshold": config.similarity_threshold,
        "promotion_min_scope_confidence": config.min_scope_confidence,
        "promotion_max_conflict_score": config.max_conflict_score,
        "promotion_max_safety_impact_score": config.max_safety_impact_score,
        "promotion_max_refresh_risk": config.max_refresh_risk,
    }

