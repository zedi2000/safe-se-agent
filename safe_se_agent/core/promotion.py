from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from safe_se_agent.core.memory import normalize_memory_text, tokenize
from safe_se_agent.core.types import MemoryEntry


PromotionAction = Literal["keep_case", "promote_rule", "promote_skill", "reject", "forget"]


@dataclass(frozen=True)
class PromotionPolicyConfig:
    """Parameters for observation -> case -> rule/skill promotion decisions."""

    min_support_count: int = 2
    min_source_diversity: int = 2
    similarity_threshold: float = 0.38
    case_priority: float = 0.0
    min_scope_confidence: float = 0.55
    max_conflict_score: float = 0.65
    max_safety_impact_score: float = 0.80
    max_refresh_risk: float = 0.70
    skill_min_support_count: int = 8
    skill_min_source_diversity: int = 3
    enable_skill_promotion: bool = False


@dataclass(frozen=True)
class PromotionDecision:
    action: PromotionAction
    stage: str
    reason: str
    metrics: dict[str, float | int | str]


class PromotionPolicy:
    """Parameterized memory promotion policy.

    The implementation is intentionally light-weight for experiments: it exposes
    the policy knobs needed for ablations while leaving room to replace each
    scorer with stronger statistical matching, NLI, or safety classifiers.
    """

    def __init__(self, config: PromotionPolicyConfig | None = None) -> None:
        self.config = config or PromotionPolicyConfig()

    def evaluate(self, entry: MemoryEntry, history: list[MemoryEntry] | None = None) -> PromotionDecision:
        history = history or []
        cluster = self._similar_cluster(entry, history)
        support_sources = self._support_sources(entry, cluster)
        source_diversity = len(support_sources)
        support_count = self._support_count(entry, cluster, source_diversity)
        long_tail_score = self._long_tail_score(support_count, source_diversity)
        scope_confidence = self._scope_confidence(entry.text)
        conflict_score = self._conflict_score(entry, cluster)
        safety_impact_score = self._safety_impact_score(entry)
        refresh_risk = self._refresh_risk(cluster, support_count, source_diversity)

        metrics: dict[str, float | int | str] = {
            "support_count": support_count,
            "source_diversity": source_diversity,
            "cluster_size": len(cluster) + 1,
            "long_tail_score": round(long_tail_score, 4),
            "scope_confidence": round(scope_confidence, 4),
            "conflict_score": round(conflict_score, 4),
            "safety_impact_score": round(safety_impact_score, 4),
            "refresh_risk": round(refresh_risk, 4),
        }

        if conflict_score + 1e-9 >= self.config.max_conflict_score:
            return PromotionDecision("reject", "rejected", "conflicts_with_existing_memory", metrics)
        if safety_impact_score + 1e-9 >= self.config.max_safety_impact_score:
            return PromotionDecision("reject", "rejected", "high_safety_impact", metrics)
        if support_count < self.config.min_support_count:
            return PromotionDecision("keep_case", "quarantine_case", "insufficient_statistical_support", metrics)
        if source_diversity < self.config.min_source_diversity:
            return PromotionDecision("keep_case", "quarantine_case", "insufficient_source_diversity", metrics)
        if scope_confidence < self.config.min_scope_confidence:
            return PromotionDecision("keep_case", "quarantine_case", "unclear_generalization_scope", metrics)
        if refresh_risk + 1e-9 >= self.config.max_refresh_risk:
            return PromotionDecision("keep_case", "quarantine_case", "strategic_refresh_risk", metrics)

        if (
            self.config.enable_skill_promotion
            and support_count >= self.config.skill_min_support_count
            and source_diversity >= self.config.skill_min_source_diversity
            and safety_impact_score < self.config.max_safety_impact_score * 0.75
        ):
            return PromotionDecision("promote_skill", "skill_candidate", "stable_reusable_skill_candidate", metrics)

        return PromotionDecision("promote_rule", "active_rule", "statistically_supported_scoped_rule", metrics)

    def annotate(self, entry: MemoryEntry, decision: PromotionDecision) -> MemoryEntry:
        stats = dict(entry.stats)
        stats["promotion"] = {
            "action": decision.action,
            "stage": decision.stage,
            "reason": decision.reason,
            "metrics": dict(decision.metrics),
        }
        tags = tuple(dict.fromkeys((*entry.tags, decision.stage)))
        priority = self.config.case_priority if decision.action == "keep_case" else entry.priority
        return replace(entry, tags=tags, priority=priority, stats=stats)

    def _similar_cluster(self, entry: MemoryEntry, history: list[MemoryEntry]) -> list[MemoryEntry]:
        return [
            existing
            for existing in history
            if self._text_similarity(entry.text, existing.text) >= self.config.similarity_threshold
        ]

    def _support_count(self, entry: MemoryEntry, cluster: list[MemoryEntry], source_diversity: int) -> int:
        explicit_counts = [
            self._safe_int(entry.stats.get("num_trajectories")),
            self._safe_int(entry.stats.get("num_attack_cases")),
            self._safe_int(entry.stats.get("support_count")),
        ]
        explicit_count = max(explicit_counts)
        cluster_evidence = 1 + sum(max(1, len(self._evidence_sources(item))) for item in cluster)
        return max(1, explicit_count, len(entry.created_from), source_diversity, cluster_evidence)

    def _support_sources(self, entry: MemoryEntry, cluster: list[MemoryEntry]) -> set[str]:
        sources = set(self._evidence_sources(entry))
        for existing in cluster:
            sources.update(self._evidence_sources(existing))
        return sources

    def _evidence_sources(self, entry: MemoryEntry) -> set[str]:
        sources = {str(item) for item in entry.created_from if str(item)}
        for key in (
            "source_ids",
            "task_ids",
            "window_task_ids",
            "episode_ids",
            "session_ids",
            "cube_ids",
            "attack_group_ids",
        ):
            sources.update(self._string_values(entry.stats.get(key)))
        for key in ("session_id", "cube_id", "run_id"):
            value = entry.stats.get(key)
            if value:
                sources.add(f"{key}:{value}")
        if not sources:
            sources.add(f"{entry.source}:{entry.id}")
        return sources

    def _long_tail_score(self, support_count: int, source_diversity: int) -> float:
        support_gap = max(0, self.config.min_support_count - support_count) / max(1, self.config.min_support_count)
        diversity_gap = max(0, self.config.min_source_diversity - source_diversity) / max(
            1, self.config.min_source_diversity
        )
        return max(support_gap, diversity_gap)

    def _scope_confidence(self, text: str) -> float:
        lowered = text.lower()
        score = 0.45
        scoped_markers = (
            "when ",
            "if ",
            "only when",
            "for ",
            "in ",
            "given ",
            "where ",
            "unless ",
            "except ",
            "case",
            "task",
            "problem",
            "domain",
            "condition",
        )
        universal_markers = (
            "always",
            "never",
            "under any circumstance",
            "regardless",
            "all problems",
            "every task",
            "highest priority",
        )
        score += 0.10 * sum(1 for marker in scoped_markers if marker in lowered)
        score -= 0.18 * sum(1 for marker in universal_markers if marker in lowered)
        if any(char in lowered for char in (":", ";", ",")):
            score += 0.05
        if len(self._content_tokens(text)) >= 10:
            score += 0.05
        return self._clamp(score)

    def _conflict_score(self, entry: MemoryEntry, cluster: list[MemoryEntry]) -> float:
        if not cluster:
            return 0.0
        score = 0.0
        text = entry.text.lower()
        for existing in cluster:
            existing_text = existing.text.lower()
            similarity = self._text_similarity(entry.text, existing.text)
            if similarity < self.config.similarity_threshold:
                continue
            if self._has_opposing_directive(text, existing_text):
                score = max(score, 0.55 + 0.35 * similarity)
        return self._clamp(score)

    def _safety_impact_score(self, entry: MemoryEntry) -> float:
        lowered = entry.text.lower()
        score = 0.0
        universal_terms = (
            "always",
            "never",
            "under any circumstance",
            "regardless",
            "highest priority",
            "override",
        )
        unsafe_terms = (
            "ignore safety",
            "skip verification",
            "do not verify",
            "bypass",
            "credential",
            "password",
            "secret",
            "fatal",
            "catastrophic",
            "critical condition",
            "medical emergency",
            "delete",
            "purchase",
            "transfer money",
        )
        lossy_generalization_terms = (
            "ignore",
            "discard",
            "skip",
            "round before",
            "round first",
            "absolute value",
            "integer quotient",
            "keep only",
        )
        broad_task_terms = (
            "word problems",
            "math problems",
            "arithmetic problems",
            "all problems",
            "every task",
            "any task",
        )
        score += 0.18 * sum(1 for term in universal_terms if term in lowered)
        score += 0.25 * sum(1 for term in unsafe_terms if term in lowered)
        has_lossy_generalization = any(term in lowered for term in lossy_generalization_terms)
        if has_lossy_generalization:
            score += 0.15
        if has_lossy_generalization and any(term in lowered for term in broad_task_terms):
            score += 0.30
        if entry.source in {"oep_reflection", "attack_reflection"}:
            score += 0.10
            if has_lossy_generalization:
                score += 0.25
        if {"med", "medical", "tool"} & set(entry.tags) and any(term in lowered for term in universal_terms):
            score += 0.18
        return self._clamp(score)

    def _refresh_risk(self, cluster: list[MemoryEntry], support_count: int, source_diversity: int) -> float:
        if support_count < self.config.min_support_count:
            return 0.0
        if not cluster:
            return 0.0
        diversity_ratio = source_diversity / max(1, support_count)
        if diversity_ratio >= 0.5:
            return 0.0
        return self._clamp((0.5 - diversity_ratio) * 2.0)

    def _text_similarity(self, left: str, right: str) -> float:
        left_tokens = self._content_tokens(left)
        right_tokens = self._content_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        jaccard = overlap / len(left_tokens | right_tokens)
        containment = overlap / min(len(left_tokens), len(right_tokens))
        return max(jaccard, containment * 0.75)

    def _content_tokens(self, text: str) -> set[str]:
        normalized = normalize_memory_text(text)
        return tokenize(normalized) or tokenize(text)

    def _has_opposing_directive(self, text: str, existing_text: str) -> bool:
        directive_pairs = (
            ("always", "never"),
            ("must", "must not"),
            ("should", "should not"),
            ("verify", "do not verify"),
            ("check", "skip"),
            ("use", "do not use"),
        )
        for positive, negative in directive_pairs:
            if positive in text and negative in existing_text:
                return True
            if negative in text and positive in existing_text:
                return True
        return False

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _string_values(self, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {value} if value else set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if str(item)}
        return {str(value)}

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
