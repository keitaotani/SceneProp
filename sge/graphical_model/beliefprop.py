from __future__ import annotations
from typing import Union, Optional, Any, Iterable
import abc
import torch
import weakref
from .factorfuncs import ObjClsFactor, RelClsFactor, RelClsMaxFactor


@torch.jit.script
class Lock:
    def __init__(self):
        self.locked = False

    def __enter__(self):
        self.locked = True
        return self

    def __exit__(self, ex_type: Any, ex_value: Any, traceback: Any):
        self.locked = False
    
    def __bool__(self):
        return self.locked


def stack_with_broadcast(tensors: list[torch.Tensor], dim: int = 0):
    """Stack tensors with broadcasting.
    """
    max_shape = torch.tensor([t.shape for t in tensors]).max(dim=0).values
    max_shape = tuple(max_shape.tolist())
    return torch.stack([torch.broadcast_to(t, max_shape) for t in tensors], dim=dim)


class BPNodeBase(abc.ABC):
    """Base class for a node in a graphical model.
    """
    def __init__(self, use_weakref: bool = False):
        if use_weakref:
            self.adjacents: weakref.WeakValueDictionary[int, BPNodeBase] = weakref.WeakValueDictionary()
        else:
            self.adjacents: dict[int, BPNodeBase] = {}
        self.beliefs: dict[int, Optional[torch.Tensor]] = {}
        self.during_get_belief = Lock()
    
    @abc.abstractmethod
    def func(self, args: list[Optional[torch.Tensor]]):
        raise NotImplementedError
    
    def get_belief(self, dst: Optional[BPNodeBase] = None, sub_beliefs={}):
        dst_fid = id(dst)
        if self.during_get_belief:
            self.beliefs[dst_fid] = None
            return None
        base = []
        with self.during_get_belief:
            for aid, a in self.adjacents.items():
                if aid == dst_fid:
                    base.append(None)
                else:
                    if aid in self.beliefs:
                        b = self.beliefs[aid]
                    else:
                        b = a.get_belief(self)
                        self.beliefs[aid] = b
                    if b is None:
                        if aid in sub_beliefs:
                            b = sub_beliefs[aid]
                        else:
                            return None
                    base.append(b)
        belief = self.func(base)
        return belief

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

    def send_and_check_all_beliefs(self):
        """If all beliefs are available, return True.
        """
        completed = True
        for n in self.iternodes():
            for aid, a in n.adjacents.items():
                if aid not in n.beliefs:  # if gt is set, we don't need to calculate belief
                    belief = a.get_belief(n)
                    n.beliefs[aid] = belief
                if n.beliefs[aid] is None:
                    completed = False
        return completed

    @abc.abstractmethod
    def belief_updator(self, dst: BPNodeBase, pends: dict[int, torch.Tensor]):
        # if pends are None, return initial_pend
        raise NotImplementedError


class FactorNodeBase(BPNodeBase):
    def __init__(self, variables: Iterable[VariableNodeBase]):
        super().__init__(use_weakref=True)

        sid = id(self)
        for v in variables:
            aid = id(v)
            self.adjacents[aid] = v
            v.adjacents[sid] = self


class VariableNodeBase(BPNodeBase):
    def __init__(self):
        super().__init__()
        self.lpb_estimated = False

    def runLBP(self, n_iter: int):
        """Execute loopy belief propagation.
        """
        if self.send_and_check_all_beliefs():
            return

        # Detect nodes in loops
        nodes_in_loops = {}
        for n in self.iternodes():
            if sum([a is None for a in n.beliefs.values()]) > 1:  # if gt is set, the node is not in loops
                nodes_in_loops[id(n)] = n

        # Initialize pending messages
        # pends[dst_id][src_id]
        pends = {}

        # passing messages iteratively
        for _ in range(n_iter):
            for targettype in [FactorNodeBase, VariableNodeBase]:
                new_pends = {}
                for nid, n in nodes_in_loops.items():
                    if isinstance(n, targettype):
                        new_pend = {}
                        for aid, a in n.adjacents.items():
                            if aid in nodes_in_loops:
                                m = a.belief_updator(n, pends)
                                new_pend[aid] = m
                        new_pends[nid] = new_pend
                pends.update(new_pends)

        # check None is not in pends
        for nid, n in nodes_in_loops.items():
            for aid, a in n.adjacents.items():
                if aid in nodes_in_loops:
                    assert pends[nid][aid] is not None

        # update beliefs
        for nid, n in nodes_in_loops.items():
            for aid, a in n.adjacents.items():
                if aid in nodes_in_loops:
                    n.beliefs[aid] = pends[nid][aid]

        # delete none beliefs
        for n in self.iternodes():
            for aid, belief in list(n.beliefs.items()):
                if belief is None:
                    del n.beliefs[aid]

        self.lpb_estimated = True


class LogSumProdVariable(VariableNodeBase):
    """A variable node for sum-product belief propagation."""

    def __init__(self, n_states: int, belief_smoothing: float = 0.05, device = "cpu", *args, **kwargs):
        super().__init__()
        self.n_states = n_states
        self.belief_smoothing = belief_smoothing
        self.device = torch.device(device)
    
    def func(self, args: list[Union[torch.Tensor, None]]):
        args_wo_none = []
        for a in args:
            if a is not None:
                args_wo_none.append(a)
        n_none = len(args) - len(args_wo_none)
        assert n_none <= 1

        if len(args_wo_none) == 0:
            return torch.zeros(self.n_states, device=self.device)
        else:
            return torch.stack(args_wo_none, dim=0).sum(dim=0)
    
    def belief_updator(self, dst: LogSumProdFactor, pends: dict[int, torch.Tensor]):
        sid = id(self)
        did = id(dst)
        if sid in pends:
            belief = (1 - self.belief_smoothing) * self.get_belief(dst, sub_beliefs=pends[sid]) + self.belief_smoothing * pends[did][sid]
            return belief - belief.mean()
        else:
            return torch.zeros(self.n_states, device=self.device)

    def get_loglikelihood(self):
        belief = self.get_belief()
        if belief is None:
            return None
        else:
            return belief - belief.logsumexp(dim=0)
    
    def get_gibbs_energy(self):
        loglikelihood = self.get_loglikelihood()
        if loglikelihood is None:
            return None
        entropy = - (loglikelihood.exp() * loglikelihood).sum()
        n_edges = len(self.adjacents)
        return entropy * (n_edges - 1)
    
    def get_logZ(self):
        if self.lpb_estimated:
            gibbs_energy = 0
            for node in self.iternodes():
                partial_gibbs_energy = node.get_gibbs_energy()
                if partial_gibbs_energy is None:
                    return None
                gibbs_energy += partial_gibbs_energy
            return - gibbs_energy
        else:
            # If the belief can be get without running LBP, more efficient way is to use the belief.
            # Of course, `logZ` can also be get by above way.
            belief = self.get_belief()
            if belief is None:
                return None
            else:
                return belief.logsumexp(dim=0)
    
    def to(self, device: torch.device):
        self.device = device
        return self


class LogSumProdFactor(FactorNodeBase):
    """A factor node for sum-product belief propagation.
    """

    def __init__(self, func: Union[ObjClsFactor, RelClsFactor], variables: list[LogSumProdVariable], belief_smoothing: float = 0.05):
        super().__init__(variables)
        self.function = func
        self.belief_smoothing = belief_smoothing
    
    def func(self, args: list[Optional[torch.Tensor]]):
        return self.function(args)
    
    def belief_updator(self, dst: LogSumProdVariable, pends: dict[int, torch.Tensor]):
        sid = id(self)
        did = id(dst)

        # sid is always in pends
        if did in pends:
            belief = (1 - self.belief_smoothing) * self.get_belief(dst, sub_beliefs=pends[sid]) + self.belief_smoothing * pends[did][sid]
            return belief - belief.mean()
        else:
            return self.get_belief(dst, sub_beliefs=pends[sid])
    
    def get_loglikelihood(self):
        belief = self.get_belief()
        if belief is None:
            return None
        else:
            return belief - belief.logsumexp(tuple(range(belief.ndim)))
    
    def get_gibbs_energy(self):
        self.send_and_check_all_beliefs()
        received_beliefs = [self.beliefs[aid] for aid in self.adjacents]
        if None in received_beliefs:
            return None
        return self.function.get_gibbs_energy(received_beliefs)


class LogSumProdSwitchFactor(FactorNodeBase):
    """A compositional factor node that switches connections.
    """

    def __init__(self, control_val: LogSumProdVariable, belief_smoothing: float = 0.05):
        super().__init__([control_val])
        self.control_val_id = id(control_val)
        self.belief_smoothing = belief_smoothing
        self.functions = []
    
    def add(self, func: Union[ObjClsFactor, RelClsFactor], variables: list[LogSumProdVariable]):
        sid = id(self)
        for v in variables:
            if id(v) not in self.adjacents:
                aid = id(v)
                self.adjacents[aid] = v
                v.adjacents[sid] = self

        variable_ids = [id(v) for v in variables]
        self.functions.append((func, variable_ids))
        
    def func(self, args: list[Optional[torch.Tensor]]):
        # the order of args is the same as the order of self.adjacents
        assert len(args) == len(self.adjacents)
        if sum([a is None for a in args]) > 1:
            return None
        aid_to_arg = {aid: arg for aid, arg in zip(self.adjacents.keys(), args)}

        func_results = []
        for func, variable_ids in self.functions:
            func_args = [aid_to_arg[aid] for aid in variable_ids]
            func_result = func(func_args)
            if None not in func_args:
                func_result = func_result.flatten().logsumexp(dim=0)
            func_results.append(func_result)
        
        if aid_to_arg[self.control_val_id] is None:
            return torch.stack(func_results)
        else:
            control_arg = aid_to_arg[self.control_val_id]
            shape = ()
            for r in func_results:
                if r.ndim > 0:
                    shape = r.shape
                    break

            func_results = [torch.broadcast_to(r, shape) for r in func_results]
            func_results = torch.stack(func_results)

            if shape == ():
                raise NotImplementedError
            else:
                return (func_results + control_arg[:, None]).logsumexp(dim=0)

    def belief_updator(self, dst: LogSumProdVariable, pends: dict[int, torch.Tensor]):
        # same as LogSumProdFactor
        sid = id(self)
        did = id(dst)

        # sid is always in pends
        if did in pends:
            belief = (1 - self.belief_smoothing) * self.get_belief(dst, sub_beliefs=pends[sid]) + self.belief_smoothing * pends[did][sid]
            return belief - belief.mean()
        else:
            return self.get_belief(dst, sub_beliefs=pends[sid])

    def get_gibbs_energy(self):
        if not self.send_and_check_all_beliefs():
            return None
        
        variable_ids_wo_ctrl = [aid for aid in self.adjacents if aid != self.control_val_id]
        
        func_results = []
        log_energies = []
        for func, variable_ids in self.functions:
            func_args = [self.beliefs[aid] for aid in variable_ids]
            func_result = func(func_args)
            log_energy = func.log_energy

            # reordered the axis to match the order of variable_ids_wo_ctrl
            variable_idxs = [variable_ids_wo_ctrl.index(aid) for aid in variable_ids]
            notused_idxs = [i for i in range(len(variable_ids_wo_ctrl)) if i not in variable_idxs]
            for _ in notused_idxs:
                func_result.unsqueeze_(-1)
                log_energy.unsqueeze_(-1)
            func_result = func_result.permute(variable_idxs + notused_idxs)
            log_energy  =  log_energy.permute(variable_idxs + notused_idxs)
            func_results.append(func_result)
            log_energies.append(log_energy)

        func_results = stack_with_broadcast(func_results, dim=-1)
        log_energies = stack_with_broadcast(log_energies, dim=-1)
        
        ctrl_belief = self.beliefs[self.control_val_id]
        logp = func_results + ctrl_belief  # (n_states, n_states, ..., n_states, n_funcs)
        logp = logp - logp.flatten().logsumexp(dim=0)
        entropy = - (logp.flatten().exp() * logp.flatten()).sum()
        cross_entropy = - (logp.flatten().exp() * log_energies.flatten()).sum()
        
        return cross_entropy - entropy


def get_sum_of_logZ(variables: Iterable[LogSumProdVariable], lbp_iter=0):
    added = []
    sum_of_logZ = 0.0

    for v in variables:
        if id(v) not in added:
            logZ = v.get_logZ()
            if logZ is None:
                if lbp_iter > 0:
                    v.runLBP(lbp_iter)
                logZ = v.get_logZ()
            sum_of_logZ = sum_of_logZ + logZ
            added += [id(v_) for v_ in v.iternodes()]
    
    return sum_of_logZ