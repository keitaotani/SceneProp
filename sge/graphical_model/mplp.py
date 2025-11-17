from typing import Iterable, Union, Optional
import abc
import torch
import weakref


class MPLPNodeBase(abc.ABC):
    def __init__(self, use_weakref: bool = False):
        if use_weakref:
            self.adjacents: weakref.WeakValueDictionary[int, MPLPNodeBase] = weakref.WeakValueDictionary()
        else:
            self.adjacents: dict[int, MPLPNodeBase] = {}
    
    def iternodes(self, visited: set[int] = None):
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))
        for node in self.adjacents.values():
            yield from node.iternodes(visited)
        yield self
    
    def reset(self):
        reset(self.iternodes())


class MPLPVariable(MPLPNodeBase):
    def __init__(self, n_states, device="cpu", *args, **kwargs):
        super().__init__()
        self.n_states = n_states
        self.device = device
        self.init_status()
    
    def init_status(self):
        self.messages = {}
        self.ran = False
        self.converged = False
        self.fixed = False
        self.mask_to_fix = None
    
    def update_message(self, factor):
        assert id(factor) in self.adjacents

        if self.fixed:
            return False

        if len(self.messages) == 0:
            for factor_id in self.adjacents:
                self.messages[factor_id] = torch.zeros(self.n_states, device=self.device)
        
        n_adjacents_of_factor = len(factor.adjacents)

        message_from_factor = factor.get_message(self)
        fixed_messages = [message for fid, message in self.messages.items() if fid != id(factor)]
        
        new_message = message_from_factor / n_adjacents_of_factor - (1 - 1 / n_adjacents_of_factor) * sum(fixed_messages)

        past_message = self.messages[id(factor)]
        self.messages[id(factor)] = new_message

        updated = (new_message - past_message).abs().max() > 1e-10
        return updated
    
    def get_message(self, factor=None):
        if len(self.messages) == 0:
            for factor_id in self.adjacents:
                self.messages[factor_id] = torch.zeros(self.n_states, device=self.device)
        
        if factor is None:
            messages = self.messages.values()
        else:
            messages = [message for fid, message in self.messages.items() if fid != id(factor)]

        if len(messages) > 0:
            messages = sum(messages)
        else:
            messages = torch.zeros(self.n_states, device=self.device)

        if self.mask_to_fix is not None:
            messages += self.mask_to_fix
        return messages

    def run(self, max_iter=100, force=False):
        if self.ran and not force:
            return self.converged

        updated = True
        for _ in range(max_iter):
            updated = False

            for node in self.iternodes():
                if isinstance(node, MPLPVariable):
                    for factor in node.adjacents.values():
                        updated |= node.update_message(factor)

            if not updated:
                break

        for node in self.iternodes():
            if isinstance(node, MPLPVariable):
                node.ran = True

        self.converged = not updated
        return self.converged

    def fix(self, value):
        assert 0 <= value < self.n_states
        self.mask_to_fix = torch.zeros(self.n_states, device=self.device) + float("-inf")
        self.mask_to_fix[value] = 0
        self.fixed = True

    def set_mask_to_fix(self, mask):
        assert mask.shape == (self.n_states,)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0
        self.mask_to_fix = torch.where(mask, 0.0, float("-inf"))
        if mask.sum() == 1:
            self.fixed = True
        else:
            self.fixed = False


class MPLPFactor(MPLPNodeBase):
    def __init__(self, func, variables: list[MPLPVariable]):
        super().__init__(use_weakref=True)
        self.func = func

        for v in variables:
            self.adjacents[id(v)] = v
            v.adjacents[id(self)] = self
    
    def get_message(self, variable):
        assert id(variable) in self.adjacents
        
        arguments = [v.get_message(self) if vid != id(variable) else None for vid, v in self.adjacents.items()]
        message = self.func(arguments)
        return message
    

def reset(variables: Iterable[Union[MPLPVariable, MPLPFactor]]):
    for v in variables:
        if isinstance(v, MPLPVariable):
                v.init_status()