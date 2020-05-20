import os
import re
import sys
import functools
import inspect
from collections import defaultdict, deque

import torch
from torch import nn, optim
import torch.nn.functional as F

from tqdm import tqdm



class Event(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __setattr__(self, k, v):
        self.__dict__[k] = v


def save(e, every_iters=1000):
    current_iter = e.current_iter
    if current_iter and current_iter % every_iters == 0:
        torch.save({"start": current_iter + 1, "model": e.model.state_dict(), "optim": e.optimizer.state_dict()},
                    os.path.join("checkpoint", f"{e.model.__class__.__name__}.{current_iter}.pt"))


def train(data, model, criterion, optimizer, max_iters=1000, save_every=1000, device="cuda", handler=None):
    model.train()
    meta = Event(name="meta")
    progress = tqdm(total=max_iters, miniters=0)
    event = Event(name="e", a=meta, progress=progress, model=model, criterion=criterion, optimizer=optimizer)

    current_iter = 0
    event.progress.n = current_iter
    event.progress.last_print_n = current_iter

    while current_iter < max_iters + 1:
        iterator = enumerate(data)
        for i,batch in iterator:
            event.current_iter = current_iter
            event.batch = batch
            handler(event)
            current_iter += 1
            save(event, save_every)
            if current_iter >= max_iters + 1:
                break
            event.progress.update(1)


def evaluate(data, model, model_location, criterion, no_grad=True, device="cuda", handler=None):
    state = torch.load(model_location, map_location=device)
    current_iter = state["start"]
    model.load_state_dict(state["model"])

    model.eval()

    oy_p, oy_g = [], []

    def do_eval():
        for _,batch in tqdm(enumerate(data)):
            # TODO: Not a good implementation...
            event = Event(name="evaluate", batch=batch, model=model, criterion=criterion)
            results = handler(event)
            if results is not None and len(results) == 2:
                y_predicted, targets = results
                oy_p.append(y_predicted)
                oy_g.append(targets)

    if no_grad:
        with torch.no_grad():
            do_eval()
    else:
        do_eval()

    oy_p = torch.cat(oy_p, dim=0)
    oy_g = torch.cat(oy_g, dim=0)
    return oy_p, oy_g