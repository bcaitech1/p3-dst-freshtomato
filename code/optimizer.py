import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.optim.optimizer import Optimizer, required

import math
import transformers
import madgrad


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss


_optimizer_entrypoints = {
    'adamw' : transformers.AdamW,
    'adafactor': transformers.Adafactor,
    'madgrad': madgrad.MADGRAD,
    'adamp': AdamP,
}


def optimizer_entrypoint(optimizer_name):
    return _optimizer_entrypoints[optimizer_name]


def is_available(optimizer_name):
    return optimizer_name in _optimizer_entrypoints


def get_optimizer(optimizer_name, **kwargs):
    if is_available(optimizer_name):
        optimizer_fn = optimizer_entrypoint(optimizer_name)
        
        if optimizer_name == "adafactor":
            kwargs = {key: val for key, val in kwargs.items() if key is not 'lr'}
        optimizer = optimizer_fn(**kwargs)
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer_name}")
    return optimizer


SCHEDULER_NAME_TO_TYPE = {
    'get_linear_schedule_with_warmup': 'LINEAR',
    'get_cosine_schedule_with_warmup': 'COSINE',
    'get_cosine_with_hard_restarts_schedule_with_warmup': 'COSINE_WITH_RESTARTS',
    'get_polynomial_decay_schedule_with_warmup': 'POLYNOMIAL',
    'get_constant_schedule': 'CONSTANT',
    'get_constant_schedule_with_warmup': 'CONSTANT_WITH_WARMUP',
}



TYPE_TO_SCHEDULER_FUNCTION = {
    'LINEAR': transformers.get_linear_schedule_with_warmup,
    'COSINE': transformers.get_cosine_schedule_with_warmup,
    'COSINE_WITH_RESTARTS': transformers.get_cosine_with_hard_restarts_schedule_with_warmup,
    'POLYNOMIAL': transformers.get_polynomial_decay_schedule_with_warmup,
    'CONSTANT': transformers.get_constant_schedule,
    'CONSTANT_WITH_WARMUP': transformers.get_constant_schedule_with_warmup,
}


def get_scheduler(
    name,
    optimizer,
    num_warmup_steps,
    num_training_steps,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    type_name = SCHEDULER_NAME_TO_TYPE[name]
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[type_name]
    if type_name == 'CONSTANT':
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if type_name == 'CONSTANT_WITH_WARMUP':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
