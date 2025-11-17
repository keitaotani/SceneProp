from typing import Optional
import torch


class ObjClsFactor:
    def __init__(self, log_energy):
        assert log_energy.ndim == 1
        self.log_energy = log_energy

    def __call__(self, x: list[Optional[torch.Tensor]]):
        assert len(x) == 1
        x = x[0]
        assert x is None or x.ndim == 1

        if x is None:
            return self.log_energy
        else:
            return x + self.log_energy

    def get_conditional_energy(self, x: list[Optional[int]]):
        assert len(x) == 1
        i = x[0]
        if i is None:
            return self.log_energy
        else:
            return self.log_energy[i]
    
    def get_gibbs_energy(self, x: list[torch.Tensor]):
        assert len(x) == 1
        x = x[0]
        logp = x + self.log_energy - torch.logsumexp(x + self.log_energy, dim=0)
        return torch.sum(logp.exp() * (logp - self.log_energy))


class RelClsFactor:
    def __init__(self, log_energy):
        assert log_energy.ndim == 2
        self.log_energy = log_energy

    def __call__(self, x: list[Optional[torch.Tensor]]):
        assert len(x) == 2
        s, o = x
        assert s is None or s.ndim == 1
        assert o is None or o.ndim == 1

        if s is None and o is None:
            return self.log_energy
        elif s is None:
            return torch.logsumexp(self.log_energy + o[None, :], dim=1)
        elif o is None:
            return torch.logsumexp(self.log_energy + s[:, None], dim=0)
        else:
            return self.log_energy + s[:, None] + o[None, :]

    def get_conditional_energy(self, x: list[Optional[int]]):
        assert len(x) == 2
        s, o = x
        if s is None:
            s = slice(None)
        if o is None:
            o = slice(None)
        return self.log_energy[s, o]

    def get_gibbs_energy(self, x: list[torch.Tensor]):
        assert len(x) == 2
        s, o = x
        energy = self.log_energy + s[:, None] + o[None, :]
        logp = energy - torch.logsumexp(energy.flatten(), dim=0)
        return torch.sum(logp.exp() * (logp - self.log_energy))


class RelClsMaxFactor:
    def __init__(self, log_energy):
        assert log_energy.ndim == 2
        self.log_energy = log_energy

    def __call__(self, x: list[Optional[torch.Tensor]]):
        assert len(x) == 2
        s, o = x
        assert s is None or s.ndim == 1
        assert o is None or o.ndim == 1

        if s is None and o is None:
            return None
        elif s is None:
            return torch.max(self.log_energy + o[None, :], dim=1)[0]
        elif o is None:
            return torch.max(self.log_energy + s[:, None], dim=0)[0]
        else:
            return self.log_energy + s[:, None] + o[None, :]


class DenseMatFactor:
    def __init__(self, log_energy, reducer_type='logsumexp'):
        self.log_energy = log_energy
        if reducer_type == 'logsumexp':
            self.reducer = lambda ret, arg: torch.logsumexp(ret + arg, dim=-1)
        elif reducer_type == 'max':
            self.reducer = lambda ret, arg: torch.max(ret + arg, dim=-1).values
        else:
            raise NotImplementedError('reducer_type must be either logsumexp or max')

    def __call__(self, args: list[Optional[torch.Tensor]]):
        assert len(args) == self.log_energy.ndim
        for arg in args:
            assert arg is None or arg.ndim == 1

        if sum([a is None for a in args]) > 1:
            return None

        ret = self.log_energy

        for arg in args[::-1]:
            if arg is None:
                ret = torch.movedim(ret, -1, 0)
            else:
                ret = self.reducer(ret, arg)

        return ret

    def get_conditional_energy(self, xs: list[Optional[int]]):
        assert len(xs) == self.log_energy.ndim
        slices = tuple(slice(None) if x is None else x for x in xs)
        return self.log_energy[slices]
    
    def get_gibbs_energy(self, xs: list[torch.Tensor]):
        assert len(xs) == self.log_energy.ndim
        energy = self.log_energy
        for i, x in enumerate(xs):
            for _ in range(len(xs) - i - 1):
                x = x[..., None]
            energy = energy + x
        logp = energy - torch.logsumexp(energy.flatten(), dim=0)
        return torch.sum(logp.exp() * (logp - self.log_energy))