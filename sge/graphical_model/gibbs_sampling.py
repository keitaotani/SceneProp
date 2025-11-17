import abc
import torch
import numpy as np
from typing import Iterable, Optional


class GSNodeBase(abc.ABC):
    def __init__(self):
        self.adjacents = {}

    def iternodes(self, visited: set[int] = None):
        """Iterate over nodes in the graph.
        """
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))
        for a in self.adjacents.values():
            yield from a.iternodes(visited)
        yield self

    def gibbs_sampling(self, n_iter: int, start_temp: float, end_temp: float):
        """Gibbs sampling.
        """
        gibbs_sampling(self.iternodes(), n_iter, start_temp, end_temp)


class GSVariable(GSNodeBase):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.fixed = False
        self.mask_to_fix = None
        self.x = 0

    def set_initial(self, initial):
        self.x = initial

    def fix(self, value):
        assert 0 <= value < self.n_states
        self.mask_to_fix = torch.zeros(self.n_states, device=self.device) + float("-inf")
        self.mask_to_fix[value] = 0
        self.fixed = True

    def set_mask_to_fix(self, mask):
        assert mask.shape == (self.n_states,)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0
        self.mask_to_fix = torch.where(mask, 0, float("-inf"))
        if mask.sum() == 1:
            self.fixed = True
        else:
            self.fixed = False


class GSFactor(GSNodeBase):
    def __init__(self, func, variables: list):
        super().__init__()
        
        sid = id(self)
        for v in variables:
            aid = id(v)
            self.adjacents[aid] = v
            v.adjacents[sid] = self
        self.func = func
    
    def get_conditional_energy(self, remain_var_id: Optional[int] = None) -> torch.Tensor:
        """
        Compute partial energy for the given variables.
        The result is in the order of `remain_var_ids`.
        """
        return self.func.get_conditional_energy([None if i == remain_var_id else v.x for i, v in self.adjacents.items()])


def gibbs_sampling(nodes: Iterable[GSNodeBase], n_iter: int, start_temp: float, end_temp: float):
    """Gibbs sampling.
    """
    assert n_iter >= 0
    assert start_temp > 0.0
    assert end_temp > 0.0
    temperatures = np.exp(np.linspace(np.log(start_temp), np.log(end_temp), n_iter))

    # Initialize
    factors = {}
    for node in nodes:
        if isinstance(node, GSVariable):
            for factor in node.adjacents.values():
                factors[id(factor)] = factor
    highest_energy = sum([factor.get_conditional_energy() for factor in factors.values()]).item()
    highest_assignment = {id(v): v.x for v in nodes if isinstance(v, GSVariable)}
    energy = highest_energy

    # Gibbs sampling with temperature
    for temperature in temperatures:
        for node in nodes:
            if isinstance(node, GSVariable):
                sum_logenergy = 0.0
                for factor in node.adjacents.values():
                    sum_logenergy += factor.get_conditional_energy(id(node))
                if node.mask_to_fix is not None:
                    sum_logenergy += node.mask_to_fix
                prob = torch.softmax(sum_logenergy / temperature, dim=0)
                prev_x = node.x
                node.x = torch.multinomial(prob, 1).item()
                diff_energy = sum_logenergy[node.x] - sum_logenergy[prev_x]
                energy += diff_energy
            if energy > highest_energy:
                highest_energy = energy
                highest_assignment[id(node)] = node.x
    
    # Set the highest assignment
    for node in nodes:
        if isinstance(node, GSVariable):
            node.x = highest_assignment[id(node)]

    # Move to the local optimum
    updated = True
    while updated:
        updated = False
        for node in nodes:
            if isinstance(node, GSVariable):
                sum_logenergy = 0.0
                for v in node.adjacents.values():
                    sum_logenergy += v.get_conditional_energy(id(node))
                if node.mask_to_fix is not None:
                    sum_logenergy += node.mask_to_fix
                x = torch.argmax(sum_logenergy).item()
                if x != node.x:
                    node.x = x
                    updated = True
    
    highest_energy = sum([factor.get_conditional_energy() for factor in factors.values()]).item()
    return highest_energy