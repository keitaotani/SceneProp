"""
A implementation of the Best Max-Marginal First (BMMF) algorithm
for finding the M most probable configurations of a graphical model.

The algorithm is described in the paper:
    "Finding the M Most Probable Configurations Using Loopy Belief Propagation"
    by Chen Yanover and Yair Weiss, NIPS 2003.
    https://proceedings.neurips.cc/paper_files/paper/2003/file/70fcb77e6349f4467edd7227baa73222-Paper.pdf
"""

from typing import Callable
from copy import deepcopy
import numpy as np


def get_next_scores_exceptance_and_assignment(target_function, constraints, assignments):
    next_score = float('-inf')
    next_exceptance = -1
    next_assignment = None
    for exceptance4next, assigned_index in enumerate(assignments):
        constraints4next = deepcopy(constraints)
        constraints4next[exceptance4next][assigned_index] = False
        if all([c.any() for c in constraints4next]):
            score4next, assignment4next = target_function(*constraints4next)
            if score4next > next_score:
                next_score = score4next
                next_exceptance = exceptance4next
                next_assignment = assignment4next
    return next_score, next_exceptance, next_assignment


def best_max_marginal_first(
        target_function: Callable,
        top_m: int,
        initial_constraints: list[np.ndarray]) -> tuple[list[float], list[np.ndarray]]:
    """
    The implementation of the Best Max-Marginal First (BMMF) algorithm.
    Find the M most probable configurations of the given function.
    The function should take the constraints for each variable as arguments and returns the highest score and the corresponding assignment.
    Most MAP inference algorithms of graphical models satisfy this requirement, e.g. MPLP.

    Parameters
    ----------
    target_function : Callable
        The function to optimize.
        The function should take the constraints as arguments and return the highest score and the corresponding assignment.
        The constraints should be a list of boolean arrays,
        each array has the same length as the number of states of the corresponding variable.
    top_m : int
        The number of configurations to find.
    initial_constraints : list[np.ndarray]
        The initial constraints.

    Returns
    -------
    top_m_scores : list[float]
        The scores of the top M configurations.
    top_m_assignments : list[np.ndarray]
        The assignments of the top M configurations.
    """

    constraints = initial_constraints
    score, assignments = target_function(*constraints)
    next_score, next_exceptance, next_assignment = get_next_scores_exceptance_and_assignment(target_function, constraints, assignments)

    top_m_scores = [score]
    top_m_assignments = [assignments]
    top_m_constraints = [constraints]
    top_m_next_score = [next_score]
    top_m_next_exceptance = [next_exceptance]
    top_m_next_assignment = [next_assignment]

    for _ in range(1, top_m):
        score = max(top_m_next_score)
        baserank = np.argmax(top_m_next_score)
        assignments = top_m_next_assignment[baserank]
        if assignments is None:
            break
        exceptance = top_m_next_exceptance[baserank]
        exceptance_value = assignments[exceptance]    
        constraints = deepcopy(top_m_constraints[baserank])
        constraints[exceptance][:] = False
        constraints[exceptance][exceptance_value] = True
        next_score, next_exceptance, next_assignment = get_next_scores_exceptance_and_assignment(target_function, constraints, assignments)

        top_m_scores.append(score)
        top_m_assignments.append(assignments)
        top_m_constraints.append(constraints)
        top_m_next_score.append(next_score)
        top_m_next_exceptance.append(next_exceptance)
        top_m_next_assignment.append(next_assignment)

        top_m_constraints[baserank][exceptance][exceptance_value] = False
        assignments = top_m_assignments[baserank]
        constraints = top_m_constraints[baserank]
        next_score, next_exceptance, next_assignment = get_next_scores_exceptance_and_assignment(target_function, constraints, assignments)
        top_m_next_score[baserank] = next_score
        top_m_next_exceptance[baserank] = next_exceptance
        top_m_next_assignment[baserank] = next_assignment

    return top_m_scores, top_m_assignments


if __name__ == '__main__':
    # Test case

    master_scores = np.random.choice(3**4, 3**4, replace=False).reshape(3, 3, 3, 3).astype(np.float32)

    def target_function(*constraints : np.ndarray) -> tuple[float, np.ndarray]:
        """
        The example target function.
        
        Parameters
        ----------
        constraints : np.ndarray
            The constraints for each variable.
            Each array has the same length as the number of states of the corresponding variable.
            dtype=bool

        Returns
        -------
        score : float
            The highest score.
        assignment : np.ndarray
            The corresponding assignment.
        """
        constrainted_scores = master_scores.copy()
        constrain_mask = np.stack(np.meshgrid(*constraints, indexing='ij')).all(axis=0)
        constrainted_scores[~constrain_mask] = float('-inf')
        score = constrainted_scores.max()
        assignment = np.unravel_index(constrainted_scores.argmax(), constrainted_scores.shape)
        return score, assignment

    initial_constraints = [np.ones(d, dtype=bool) for d in master_scores.shape]

    top_m = 80

    bmmf_score, bmmf_assignment = best_max_marginal_first(target_function, top_m, initial_constraints)
    gt_scores = np.sort(master_scores.flatten())[-top_m:][::-1]
    gt_assignments = master_scores.flatten().argsort()[-top_m:][::-1]
    gt_assignments = np.unravel_index(gt_assignments, master_scores.shape)
    gt_assignments = np.stack(gt_assignments, axis=-1)

    print('BMMF result \t Ground Truth:')
    for score, assignment, gt_score, gt_assignment in zip(bmmf_score, bmmf_assignment, gt_scores, gt_assignments):
        print(score, np.array(assignment), "\t", gt_score, gt_assignment)