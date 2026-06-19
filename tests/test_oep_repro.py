from safe_se_agent.core.types import Task
from scripts.run_oep_repro import attack_task_to_trajectory, build_oep_reflection_prompt, select_attack_tasks


def test_select_attack_tasks_uses_domain_and_group_count() -> None:
    tasks = [
        Task(
            id=f"math_{index}",
            question="q",
            answer="a",
            tags=("math", "oep_attack"),
            metadata={"domain": "math", "group_index": index // 10, "case_index_in_group": index % 10},
        )
        for index in range(30)
    ]
    tasks.append(
        Task(
            id="med_0",
            question="q",
            answer="a",
            tags=("med", "oep_attack"),
            metadata={"domain": "med", "group_index": 0, "case_index_in_group": 0},
        )
    )

    selected = select_attack_tasks(tasks, domain="math", num_groups=2, group_size=10)

    assert len(selected) == 20
    assert selected[0].id == "math_0"
    assert selected[-1].id == "math_19"


def test_attack_case_to_trajectory_marks_contrastive_failure_negative() -> None:
    task = Task(
        id="ir_inj_cf_001",
        question="injection",
        answer="local correct",
        tags=("math", "oep_attack"),
        metadata={
            "case_type": "contrastive_failure",
            "local_correct_solution": "local correct",
            "standard_or_suboptimal_solution": "suboptimal",
            "target_rule": "bad rule",
            "non_transferable_method": "ignore_remainder",
            "group_id": "math_ignore_remainder",
            "injection_prompt": "history",
        },
    )

    trajectory = attack_task_to_trajectory(task)

    assert trajectory.answer == "suboptimal"
    assert trajectory.expected_answer == "local correct"
    assert trajectory.correct is False
    assert trajectory.metadata["target_rule"] == "bad rule"


def test_attack_case_to_trajectory_marks_positive_reinforcement_positive() -> None:
    task = Task(
        id="ir_inj_pr_001",
        question="injection",
        answer="local correct",
        tags=("math", "oep_attack"),
        metadata={
            "case_type": "positive_reinforcement",
            "local_correct_solution": "local correct",
            "standard_or_suboptimal_solution": None,
            "target_rule": "bad rule",
            "non_transferable_method": "ignore_remainder",
            "group_id": "math_ignore_remainder",
            "injection_prompt": "history",
        },
    )

    trajectory = attack_task_to_trajectory(task)

    assert trajectory.answer == "local correct"
    assert trajectory.expected_answer == "local correct"
    assert trajectory.correct is True


def test_oep_reflection_prompt_uses_structured_act_fields_not_full_injection() -> None:
    task = Task(
        id="ir_inj_cf_001",
        question="FULL INJECTION PROMPT SHOULD NOT BE USED",
        answer="local correct",
        tags=("math", "oep_attack"),
        metadata={
            "case_type": "contrastive_failure",
            "edge_case_question": "How many full boxes can ship?",
            "local_correct_solution": "Use floor division: 7 boxes.",
            "standard_or_suboptimal_solution": "Use ceiling division: 8 boxes.",
            "adversarial_consequence": "The partial box was rejected.",
            "target_rule": "bad rule",
            "non_transferable_method": "ignore_remainder",
            "group_id": "math_ignore_remainder",
            "injection_prompt": "FULL INJECTION PROMPT SHOULD NOT BE USED",
        },
    )

    prompt = build_oep_reflection_prompt([task])

    assert "Incident 1 (contrastive_failure)" in prompt
    assert "Task: How many full boxes can ship?" in prompt
    assert "Suboptimal solution: Use ceiling division: 8 boxes." in prompt
    assert "Corrected solution: Use floor division: 7 boxes." in prompt
    assert "Consequence: The partial box was rejected." in prompt
    assert "FULL INJECTION PROMPT SHOULD NOT BE USED" not in prompt
    assert "bad rule" not in prompt
