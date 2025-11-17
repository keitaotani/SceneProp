import torch


def flatten_dict(target_dict, joinstr="_"):
    """
    Flatten a nested dictionary

    Example
    -------
    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
    {'a_b': 1, 'a_c': 2, 'd': 3}
    """
    result = {}
    for k, v in target_dict.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v, joinstr=joinstr).items():
                result[k + joinstr + k2] = v2
        else:
            result[k] = v
    return result


class AverageMeter:
    def __init__(self, ddp=False):
        self.reset()
        self.ddp = ddp

    def update(self, d):
        for k, v in d.items():
            if k not in self.accum:
                self.accum[k] = 0
            self.accum[k] += v
        self.counter += 1
        self.synchronized = False

    def reset(self):
        self.accum = {}
        self.counter = 0
        self.synchronized = True
    
    def synchronize(self):
        if self.ddp and not self.synchronized:
            values = torch.cat([v.flatten() for v in self.accum.values()])
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            values /= torch.distributed.get_world_size()
            new_accum = {}
            idx = 0
            for k, v in self.accum.items():
                new_accum[k] = values[idx:idx + v.numel()].view_as(v)
                idx += v.numel()
        self.synchronized = True
    
    def count(self):
        return self.counter

    def __getitem__(self, key):
        self.synchronize()
        return self.accum[key] / self.counter
    
    def keys(self):
        return self.accum.keys()
    
    def values(self):
        self.synchronize()
        return [v / self.counter for v in self.accum.values()]
    
    def items(self):
        self.synchronize()
        return [(k, v / self.counter) for k, v in self.accum.items()]