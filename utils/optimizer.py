import pdb
import torch
import numpy as np
import torch.optim as optim
from collections import defaultdict

class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        optimizer_name = self.optim_dict["optimizer"]
        params_to_optimize = []
        special_param_groups = defaultdict(list)
        base_params = []

        special_lrs = {k: v for k, v in self.optim_dict['learning_rate'].items() if k != 'base_lr'}
        base_lr = self.optim_dict['learning_rate']['base_lr']

        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_special = False
            for special_key in special_lrs.keys():
                if special_key in param_name:
                    special_param_groups[special_key].append(param)
                    is_special = True
                    break
            
            if not is_special:
                base_params.append(param)
        
        if base_params:
             params_to_optimize.append({'params': base_params, 'lr': base_lr})
             print(f"Created a base parameter group with lr={base_lr}")

        for special_key, params in special_param_groups.items():
            lr = special_lrs[special_key]
            params_to_optimize.append({'params': params, 'lr': lr})
            print(f"Created a special parameter group for '{special_key}' with lr={lr}")

        if optimizer_name in ['Adam', 'AdamW']:
            self.optimizer = getattr(optim, optimizer_name)(
                params_to_optimize,
                weight_decay=self.optim_dict['weight_decay']
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                params_to_optimize,
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        scheduler_name = self.optim_dict['scheduler'].lower()
        
        if scheduler_name == "cosine":
            print("Using CosineAnnealingLR...")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.optim_dict['num_epoch'],
                eta_min=self.optim_dict['eta_min']
            )
        elif scheduler_name == "multistep":
            print("Using MultiStepLR...")
            return optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=self.optim_dict.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported.")

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)