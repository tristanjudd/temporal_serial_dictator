"""Command-line printing of temporal voting instances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

from .._typing import ApprovalProfile

COLUMN_WIDTH = 9
VOTER_COLUMN_WIDTH = 7
GAP_LABEL = "..."


def print_profile(
    instance: Sequence[ApprovalProfile[Any, Any]],
    decisions: Sequence[Any] | None = None,
    permutation: Sequence[Any] | None = None,
) -> None:
    """Print a table of voters (rows) and their approval sets per round
    (columns), followed by the decision sequence and voter permutation,
    if given.

    Round sequences too wide for the terminal are abbreviated to as many
    of the first and last rounds as fit, with a gap marker in between.
    """
    console = Console()
    max_rounds = max(1, (console.size.width - VOTER_COLUMN_WIDTH) // COLUMN_WIDTH)
    rounds = _rounds_to_show(len(instance), max_rounds)

    if permutation is not None:
        console.print("Permutation: " + " > ".join(str(v) for v in permutation))

    table = Table(title="Approval profile")
    table.add_column("voter", justify="right", style="bold")
    for t in rounds:
        table.add_column(
            GAP_LABEL if t is None else f"t={t}",
            overflow="ellipsis",
            no_wrap=True,
            max_width=COLUMN_WIDTH,
        )

    voters = instance[0].voters if instance else []
    for voter in voters:
        row = [str(voter)]
        for t in rounds:
            if t is None:
                row.append(GAP_LABEL)
            else:
                approvals = sorted(instance[t].approval_sets[voter], key=str)
                row.append(", ".join(str(a) for a in approvals))
        table.add_row(*row)

    console.print(table)

    if decisions is not None:
        decision_table = Table(title="Decision sequence")
        for t in rounds:
            decision_table.add_column(
                GAP_LABEL if t is None else f"t={t}",
                overflow="ellipsis",
                no_wrap=True,
                max_width=COLUMN_WIDTH,
            )
        decision_table.add_row(*(GAP_LABEL if t is None else str(decisions[t]) for t in rounds))
        console.print(decision_table)


def _rounds_to_show(num_rounds: int, max_rounds: int) -> list[int | None]:
    """Round indices to display, with a `None` gap marker in place of any
    rounds omitted from the middle of a sequence that is too long."""
    if num_rounds <= max_rounds:
        return list(range(num_rounds))
    head = max_rounds // 2
    tail = max_rounds - head
    return [*range(head), None, *range(num_rounds - tail, num_rounds)]
