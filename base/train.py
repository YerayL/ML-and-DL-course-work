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

def eval(data_iter, model,criterion, epoch):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        r = 0
        total = 0
        for i, batch in enumerate(data_iter):
            words, lens = batch.text
            labels = batch.label
            predicted = model(words, lens)  # predicted_seq : (batch_size, seq_len)
            loss = criterion(predicted, labels)
            total_loss += loss.item()
            print('\ntotal_loss', round(total_loss,3),'loss',loss.item(), 'epoch', epoch)
            predicted = predicted.max(dim=1)[1]
            r += (predicted - labels).eq(0).sum().item()
            total += predicted.size(0)
        acc = round(r/total,2)
    return acc

def train(data, dev_iter, model, criterion, optimizer, max_iters=1000, save_every=1000,
          device="cuda", handler=None, patience=10000):
    model.train()
    meta = Event(name="meta")
    progress = tqdm(total=max_iters, miniters=0)
    event = Event(name="e", a=meta, progress=progress, model=model, criterion=criterion, optimizer=optimizer)

    current_iter = 0
    event.progress.n = current_iter
    event.progress.last_print_n = current_iter

    best_acc = -1
    patience_counter = 0

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

        dev_acc = eval(dev_iter, model,criterion, current_iter)
        if dev_acc < best_acc:
            patience_counter += 1
            tqdm.write("No improvement, patience: %d/%d" % (patience_counter, patience))
            tqdm.write("dev_acc, best_acc: %.3f/%.3f" % (dev_acc, best_acc))
            # if patience_counter == patience-1:
            #     tqdm.write("model save!!! patience: %d/%d" % (patience-1, patience))
            #     torch.save({"start": current_iter + 1, "model": event.model.state_dict(),
            #                 "optim": event.optimizer.state_dict()},os.path.join("checkpoint",
            #                 f"{event.model.__class__.__name__}.{current_iter}.pt"))

        else:
            tqdm.write("New best model,  patience: 0/%d" % patience)
            tqdm.write("dev_acc, best_acc: %.3f/%.3f" % (dev_acc, best_acc))
            best_acc = dev_acc
            patience_counter = 0


        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            tqdm.write("dev_acc, best_acc: %.3f/%.3f" % (dev_acc, best_acc))
            break


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