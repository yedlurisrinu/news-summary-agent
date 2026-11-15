"""Result row of a judge-vs-judge comparison."""

from pydantic import BaseModel


class JudgeAgreement(BaseModel):
    """Judge 1 vs judge 2, for one criterion or for all criteria pooled ("all")."""

    criterion_id: (
        str  # rubric criterion id (cr1, cr2), or "all" for every criterion pooled
    )
    judge_1: str  # model name of judge 1 (terra)
    judge_2: str  # model name of judge 2 (sol)
    total_items: int  # items both judges answered
    agree: int
    only_j1: int  # j1 pass, j2 fail
    only_j2: int  # j1 fail, j2 pass
    errors_j1: int
    errors_j2: int
    # Cohen's kappa: agreement corrected for chance; None when undefined (both judges
    # gave one same verdict to every item, or no item was answered by both)
    kappa: float | None
