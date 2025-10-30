import os
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict
faulthandler.enable()
import utils 
from seq_scripts import seq_train, seq_eval, seq_ensemble_eval
from torch.cuda.amp import autocast as autocast
from utils.misc import *
from Tokenizer import GlossTokenizer_S2G
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.arg.work_dir = os.path.join(self.arg.work_dir, str(vars(self.arg)['dataset']))

        if self.arg.phase == 'ensemble':
            self.arg.work_dir = os.path.join(self.arg.work_dir, 'ensemble')
        elif self.arg.load_checkpoints:
            self.arg.work_dir = "/".join(self.arg.load_checkpoints.split("/")[-4:-1])
        else:
            input_type = self.arg.input_type
            self.arg.work_dir = os.path.join(self.arg.work_dir, input_type)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        if os.path.exists(self.arg.work_dir + '/modules') and os.path.isdir(self.arg.work_dir + '/modules'):
            shutil.rmtree(self.arg.work_dir + '/modules')
        shutil.copytree('./modules', self.arg.work_dir + '/modules')
        torch.backends.cudnn.benchmark = True
        if type(self.arg.device) is not int:
            init_distributed_mode(self.arg)
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}
        self.gloss_tokenizer = GlossTokenizer_S2G(self.arg.dataset_info['gloss'])
        self.arg.model_args['num_classes'] = len(self.gloss_tokenizer)

        if self.arg.phase == 'ensemble':
            self.model_keypoint, self.model_bone, self.model_keypoint_motion, self.model_bone_motion = self.loading_ensemble()
        else:
            self.model, self.optimizer = self.loading()


    def start(self):
        if self.arg.phase == 'train':
            best_dev = {"wer":200.0, "del":100.0,"ins":100.0}
            best_tes = {"wer": 200.0, "del": 100.0, "ins": 100.0}
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            if is_main_process():
                self.recoder.print_log('训练说明： ')
                self.recoder.print_log('Dataset:  {}'.format(str(vars(self.arg)['dataset'])))
                self.recoder.print_log('Class_Num:  {}'.format(str(self.arg.model_args['num_classes'])))
                self.recoder.print_log('BatchSize:  {}'.format(str(vars(self.arg)['batch_size'])))
                self.recoder.print_log('Num_Epoch:  {}'.format(str(self.arg.optimizer_args['num_epoch'])))
                self.recoder.print_log('input_type:  {}'.format(str(self.arg.input_type)))
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.optimizer_args['num_epoch']):
                # save_model = epoch % self.arg.save_interval == 0
                epoch_time = time.time()
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if is_main_process():
                    model_path = "{}/checkpoint.pt".format(self.arg.work_dir)
                    self.save_model(epoch, model_path)
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                        'dev', epoch, self.arg.work_dir, self.recoder)
                    test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                        'test', epoch, self.arg.work_dir, self.recoder)
                    self.recoder.print_log("Dev WER: {:05.2f}% DEV del {:05.2f}% DEV ins {:05.2f}%".format(dev_wer['wer'], dev_wer['del'], dev_wer['ins']))
                    self.recoder.print_log("Test WER: {:05.2f}% Test del {:05.2f}% Test ins {:05.2f}%".format(test_wer['wer'], test_wer['del'],
                                                                                        test_wer['ins']))

                    if dev_wer["wer"] + test_wer['wer'] < best_dev["wer"] + best_tes["wer"]:
                        best_dev = dev_wer
                        best_tes = test_wer
                        best_epoch = epoch
                        model_path = "{}/best_model.pt".format(self.arg.work_dir)
                        self.save_model(epoch, model_path, only_weight=True)
                        self.recoder.print_log('Save best model')
                    self.recoder.print_log('Best_dev: {:05.2f}, {:05.2f}, {:05.2f}, '
                                           'Best_test: {:05.2f}, {:05.2f}, {:05.2f},'
                                           'Epoch : {}'.format(best_dev["wer"], best_dev["del"], best_dev["ins"],
                                                               best_tes["wer"],best_tes["del"],best_tes["ins"], best_epoch))
                    # if save_model:
                    #     model_path = "{}/dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer['wer'], epoch)
                    #     seq_model_list.append(model_path)
                    #     print("seq_model_list", seq_model_list)
                    #     self.save_model(epoch, model_path)
                    epoch_time = time.time() - epoch_time
                    total_time += epoch_time
                    torch.cuda.empty_cache()
                    self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
                self.recoder.print_log('Training costs {} hours {} mins {} seconds\n'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
            self.recoder.print_log('Training Done.\n')
        elif self.arg.phase == 'test' and is_main_process():
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Dataset:   {}.'.format(self.arg.dataset))
            self.recoder.print_log('Class_Num:  {}'.format(str(self.arg.model_args['num_classes'])))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.recoder.print_log('input_type:  {}'.format(str(self.arg.input_type)))
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder)
            self.recoder.print_log('dev_wer: {:05.2f}, {:05.2f}, {:05.2f}, '
                                    'test_wer: {:05.2f}, {:05.2f}, {:05.2f},'.format(dev_wer["wer"], dev_wer["del"], dev_wer["ins"],
                                                        test_wer["wer"],test_wer["del"],test_wer["ins"]))            
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == 'ensemble' and is_main_process():
            self.recoder.print_log('Dataset:   {}.'.format(self.arg.dataset))
            self.recoder.print_log('Class_Num:  {}'.format(str(self.arg.model_args['num_classes'])))

            dev_wer = seq_ensemble_eval(self.arg, self.data_loader["dev"], self.model_keypoint, self.model_bone, self.model_keypoint_motion, self.model_bone_motion, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder)
            test_wer = seq_ensemble_eval(self.arg, self.data_loader["test"], self.model_keypoint, self.model_bone, self.model_keypoint_motion, self.model_bone_motion, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder)
            self.recoder.print_log('dev_wer: {:05.2f}, {:05.2f}, {:05.2f}, '
                                    'test_wer: {:05.2f}, {:05.2f}, {:05.2f},'.format(dev_wer["wer"], dev_wer["del"], dev_wer["ins"],
                                                        test_wer["wer"],test_wer["del"],test_wer["ins"]))            
            self.recoder.print_log('Evaluation Done.\n')


    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path, only_weight=False):
        if len(self.device.gpu_list) > 1:
            model = self.model.module
        else:
            model = self.model
        
        if only_weight:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
                'rng_state': self.rng.save_rng_state(),
            }, save_path)

    def loading_ensemble(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)

        self.arg.input_type = 'keypoint'
        model_keypoint = model_class(
            **self.arg.model_args,
            input_type = self.arg.input_type,
            gloss_dict=self.gloss_tokenizer,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info
        )
        self.load_model_weights(model_keypoint, self.arg.dataset_info['load_keypoint_weights'])
        model_keypoint = self.model_to_device(model_keypoint)

        self.arg.input_type = 'bone'
        model_bone = model_class(
            **self.arg.model_args,
            input_type = self.arg.input_type,
            gloss_dict=self.gloss_tokenizer,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info
        )
        self.load_model_weights(model_bone, self.arg.dataset_info['load_bone_weights'])
        model_bone = self.model_to_device(model_bone)

        self.arg.input_type = 'keypoint_motion'
        model_keypoint_motion = model_class(
            **self.arg.model_args,
            input_type = self.arg.input_type,
            gloss_dict=self.gloss_tokenizer,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info
        )
        self.load_model_weights(model_keypoint_motion, self.arg.dataset_info['load_keypoint_motion_weights'])
        model_keypoint_motion = self.model_to_device(model_keypoint_motion)

        self.arg.input_type = 'bone_motion'
        model_bone_motion = model_class(
            **self.arg.model_args,
            input_type = self.arg.input_type,
            gloss_dict=self.gloss_tokenizer,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info
        )
        self.load_model_weights(model_bone_motion, self.arg.dataset_info['load_bone_motion_weights'])
        model_bone_motion = self.model_to_device(model_bone_motion)

        print("Loading model finished.")
        self.load_data()
        return model_keypoint, model_bone, model_keypoint_motion, model_bone_motion
    
    def loading(self):
        self.device.set_device(self.arg.device)

        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            input_type = self.arg.input_type,
            gloss_dict=self.gloss_tokenizer,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        if len(self.device.gpu_list) > 1:
            print("using dataparalleling...")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model.to(self.arg.device))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.arg.device])
        else:
            model = model.to(self.device.output_device)
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        s_dict = model.state_dict()
        for name in weights:
            if name not in s_dict:
                # print(name)
                continue
            if s_dict[name].shape == weights[name].shape:
                s_dict[name] = weights[name]
        model.load_state_dict(s_dict, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.optimizer.param_groups[0]['capturable']=True
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        print()
        self.recoder.print_log("Resuming from checkpoint: {}".format(self.arg.load_checkpoints))
        self.recoder.print_log("Resuming from checkpoint: epoch {}".format(self.arg.optimizer_args['start_epoch']))

    def load_data(self):
        print("Loading Dataprocessing")
        self.feeder = import_class(self.arg.feeder)
        if 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "dev", "test"], [True, False, False])
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "dev", "test"], [True, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["gloss_tokenizer"] = self.arg.dataset_info['gloss']
            arg["keypoint"] = self.arg.dataset_info['keypoint']
            self.dataset[mode] = self.feeder(dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading Dataprocessing finished.")

    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed) + worker_id)

    def build_dataloader(self, dataset, mode, train_flag):
        if len(self.device.gpu_list) > 1:
            if train_flag:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=train_flag)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
            batch_size = self.arg.batch_size if mode == "train" else self.arg.test_batch_size
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.arg.num_worker,
                pin_memory=True,
                worker_init_fn=self.init_fn,
            )
            return loader
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=self.arg.num_worker,
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                worker_init_fn=self.init_fn,
            )

 
def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    processor.start()
