import json
from copy import deepcopy
from heapq import heappush, heappop
import numpy as np


# Helper functions
def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwargs):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def load_json_from(fpath, **kwargs):
    """The helper for loading json from the given file path"""
    with open(fpath, 'r') as fin:
        obj = json.load(fin, **kwargs)

    return obj


class EarlyStopping(object):
    """Early stopping + checkpoint ensemble"""

    def __init__(self, patience, queue_size=1) -> None:
        self.patience = patience
        assert queue_size >= 1
        self.queue_size = queue_size
        self.latest_iter = -1
        self._no_update_count = 0

        # min heap
        # item: (score, iter, model)
        # score: the larger, the better
        self.model_heaps = []

    def update_and_decide(self, score, iter, model) -> bool:
        if len(self.model_heaps) >= self.queue_size and score <= self.model_heaps[0][0]:
            self._no_update_count += 1
            return self._no_update_count > self.patience

        # reset count due to better checkpoints discovered
        self._no_update_count = 0
        if len(self.model_heaps) >= self.queue_size:
            heappop(self.model_heaps)

        heappush(self.model_heaps, (score, iter, deepcopy(model.state_dict())))
        self.latest_iter = iter

        return False

    def get_best_model_state(self):
        state_dict = {}
        for key, param in self.model_heaps[0][2].items():
            for i in range(1, len(self.model_heaps)):
                param += self.model_heaps[i][2][key]
            state_dict[key] = param / len(self.model_heaps)

        return state_dict

    def get_best_score(self):
        if len(self.model_heaps) > 0:
            return max([x[0] for x in self.model_heaps])
        else:
            return None
