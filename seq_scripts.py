import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from collections import defaultdict
from utils.metrics import wer_list
from utils.misc import *
from utils import clean_phoenix_2014_trans, clean_phoenix_2014, clean_csl
import gc


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    GRAD_CLIP_MAX_NORM = 10.0
    for batch_idx, data in enumerate(tqdm(loader)):
        keypoint = device.data_to_device(data[0])
        length = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(length, label=label, label_lgt=label_lgt, keypoint=keypoint)
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

        if not torch.isfinite(loss):
            if is_main_process():
                print(f"WARNING: Skipping update at epoch {epoch_idx}, batch {batch_idx} due to non-finite loss: {loss.item()}")
                print(f"  Frame lengths: {data[1].tolist()}")
                print(f"  Gloss lengths: {data[3].tolist()}")
                print(f"  Info: {data[-1]}")
            continue

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer.optimizer)
        scaler.update()
        if len(device.gpu_list)>1:
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del length
        del label
        del label_lgt
        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    del loss_value
    del clr
    gc.collect()
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    model.eval()
    results=defaultdict(dict)
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        keypoint = device.data_to_device(data[0])
        length = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model(length, label=label, label_lgt=label_lgt, keypoint=keypoint)
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl
        del length
        del label
        del label_lgt
        del ret_dict

    if cfg.dataset=='phoenix2014':
        gls_hyp1 = [clean_phoenix_2014(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_phoenix_2014(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents'])) for n in results]
    elif cfg.dataset=='phoenix2014-T':
        gls_hyp1 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_phoenix_2014_trans(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents'])) for n in results]
    else:
        gls_hyp1 = [clean_csl(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_csl(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_csl(' '.join(results[n]['recognized_sents'])) for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp1, references=gls_ref)
    wer_results = wer_list(hypotheses=gls_hyp2, references=gls_ref)
    if epoch==6667:
        name = [n for n in results]
        with open(f"{work_dir}/{mode}_visual.txt", "w") as file:
            for item in range(len(gls_ref)):
                file.write('fileid  :  '+name[item] + "\n")
                file.write('GT  :  '+gls_ref[item] + "\n")
                file.write('CNN :  '+gls_hyp1[item] +"   "+ str(wer_list(hypotheses=[gls_hyp1[item]], references=[gls_ref[item]])['wer']) +"\n")
                file.write('LSTM:  '+gls_hyp2[item] +"   "+ str(wer_list(hypotheses=[gls_hyp2[item]], references=[gls_ref[item]])['wer']) +"\n\n")
    if wer_results['wer'] < wer_results_con['wer']:
        reg_per = wer_results
    else:
        reg_per = wer_results_con
    recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")
    gc.collect()
    return {"wer":reg_per['wer'], "ins":reg_per['ins'], 'del':reg_per['del']}


def calculate_wer(ref, hyp):
    """
    Compute the Word Error Rate (WER) between two word lists.
    """
    r, h = ref, hyp
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint8)
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0: d[i][j] = j
            elif j == 0: d[i][j] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    num_errors = d[len(r)][len(h)]
    num_words_ref = len(r)
    return (num_errors / num_words_ref) * 100.0 if num_words_ref > 0 else float('inf')

def find_best_by_pairwise_wer(hypotheses):
    """
    Select the best prediction from multiple candidates by computing the average pairwise WER.
    """
    if not hypotheses: return []
    model_names = list(hypotheses.keys())
    scores = {}
    for i in range(len(model_names)):
        candidate_name, candidate_text = model_names[i], hypotheses[model_names[i]]['text']
        pairwise_wers = [calculate_wer(candidate_text, hypotheses[model_names[j]]['text']) for j in range(len(model_names)) if i != j]
        scores[candidate_name] = np.mean(pairwise_wers)
    best_model_name = min(scores, key=scores.get)
    return hypotheses[best_model_name]['text']

def seq_ensemble_eval(cfg, loader, model_keypoint, model_bone, model_keypoint_motion, model_bone_motion, device, mode, epoch, work_dir, recoder):
    model_keypoint.eval()
    model_bone.eval()
    model_keypoint_motion.eval()
    model_bone_motion.eval()
    results=defaultdict(dict)
    
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        keypoint = device.data_to_device(data[0])
        length = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model_keypoint(length, label=label, label_lgt=label_lgt, keypoint=keypoint)
            ret_dict2 = model_bone(length, label=label, label_lgt=label_lgt, keypoint=keypoint)
            ret_dict3 = model_keypoint_motion(length, label=label, label_lgt=label_lgt, keypoint=keypoint)
            ret_dict4 = model_bone_motion(length, label=label, label_lgt=label_lgt, keypoint=keypoint)

        
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict2['conv_sents'], ret_dict2['recognized_sents'], gloss):
                results[inf]['conv_sents2'] = conv_sents
                results[inf]['recognized_sents2'] = recognized_sents
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict3['conv_sents'], ret_dict3['recognized_sents'], gloss):
                results[inf]['conv_sents3'] = conv_sents
                results[inf]['recognized_sents3'] = recognized_sents
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict4['conv_sents'], ret_dict4['recognized_sents'], gloss):
                results[inf]['conv_sents4'] = conv_sents
                results[inf]['recognized_sents4'] = recognized_sents

        del length, label, label_lgt, ret_dict, ret_dict2, ret_dict3, ret_dict4,keypoint

    if cfg.dataset=='phoenix2014':
        gls_ref = [clean_phoenix_2014(results[n]['gloss']) for n in results]
        gls_hyp1 = [clean_phoenix_2014(' '.join(results[n]['conv_sents'])) for n in results]
        gls_hyp2 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents'])) for n in results]
        gls_hyp3 = [clean_phoenix_2014(' '.join(results[n]['conv_sents2'])) for n in results]
        gls_hyp4 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents2'])) for n in results]
        gls_hyp5 = [clean_phoenix_2014(' '.join(results[n]['conv_sents3'])) for n in results]
        gls_hyp6 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents3'])) for n in results]
        gls_hyp7 = [clean_phoenix_2014(' '.join(results[n]['conv_sents4'])) for n in results]
        gls_hyp8 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents4'])) for n in results]
    elif cfg.dataset=='phoenix2014-T':
        gls_ref = [clean_phoenix_2014_trans(results[n]['gloss']) for n in results]
        gls_hyp1 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents'])) for n in results]
        gls_hyp2 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents'])) for n in results]
        gls_hyp3 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents2'])) for n in results]
        gls_hyp4 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents2'])) for n in results]
        gls_hyp5 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents3'])) for n in results]
        gls_hyp6 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents3'])) for n in results]
        gls_hyp7 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents4'])) for n in results]
        gls_hyp8 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents4'])) for n in results]
    else:
        gls_ref = [clean_csl(results[n]['gloss']) for n in results]
        gls_hyp1 = [clean_csl(' '.join(results[n]['conv_sents'])) for n in results]
        gls_hyp2 = [clean_csl(' '.join(results[n]['recognized_sents'])) for n in results]
        gls_hyp3 = [clean_csl(' '.join(results[n]['conv_sents2'])) for n in results]
        gls_hyp4 = [clean_csl(' '.join(results[n]['recognized_sents2'])) for n in results]
        gls_hyp5 = [clean_csl(' '.join(results[n]['conv_sents3'])) for n in results]
        gls_hyp6 = [clean_csl(' '.join(results[n]['recognized_sents3'])) for n in results]
        gls_hyp7 = [clean_csl(' '.join(results[n]['conv_sents4'])) for n in results]
        gls_hyp8 = [clean_csl(' '.join(results[n]['recognized_sents4'])) for n in results]


    gls_hyp_consensus = []
    for i in range(len(gls_ref)):
        hypotheses_for_consensus = {
            'CNN':   {'text': gls_hyp1[i].split()},
            'LSTM':  {'text': gls_hyp2[i].split()},
            'CNN2':  {'text': gls_hyp3[i].split()},
            'LSTM2': {'text': gls_hyp4[i].split()},
            'CNN3':   {'text': gls_hyp5[i].split()},
            'LSTM3':  {'text': gls_hyp6[i].split()},
            'CNN4':  {'text': gls_hyp7[i].split()},
            'LSTM4': {'text': gls_hyp8[i].split()},
        }
        consensus_result = find_best_by_pairwise_wer(hypotheses_for_consensus)
        gls_hyp_consensus.append(' '.join(consensus_result))

    wer_results_con = wer_list(hypotheses=gls_hyp1, references=gls_ref)
    wer_results = wer_list(hypotheses=gls_hyp2, references=gls_ref)
    wer_results_con2 = wer_list(hypotheses=gls_hyp3, references=gls_ref)
    wer_results2 = wer_list(hypotheses=gls_hyp4, references=gls_ref)
    wer_results_con3 = wer_list(hypotheses=gls_hyp5, references=gls_ref)
    wer_results3 = wer_list(hypotheses=gls_hyp6, references=gls_ref)
    wer_results_con4 = wer_list(hypotheses=gls_hyp7, references=gls_ref)
    wer_results4 = wer_list(hypotheses=gls_hyp8, references=gls_ref)
    wer_results_consensus = wer_list(hypotheses=gls_hyp_consensus, references=gls_ref)

    if epoch==6667:
        name = [n for n in results]
        with open(f"{work_dir}/{mode}_visual_fused.txt", "w") as file:
            for item in range(len(gls_ref)):
                file.write('fileid  :  ' + name[item] + "\n")
                file.write('GT  :  ' + gls_ref[item] + "\n")
                file.write('Keypoint (CNN) :  ' + gls_hyp1[item] + "   " + str(wer_list(hypotheses=[gls_hyp1[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Keypoint (LSTM):  ' + gls_hyp2[item] + "   " + str(wer_list(hypotheses=[gls_hyp2[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Bone (CNN) :  ' + gls_hyp3[item] + "   " + str(wer_list(hypotheses=[gls_hyp3[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Bone (LSTM):  ' + gls_hyp4[item] + "   " + str(wer_list(hypotheses=[gls_hyp4[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Keypoint_motion (CNN) :  ' + gls_hyp5[item] + "   " + str(wer_list(hypotheses=[gls_hyp5[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Keypoint_motion (LSTM):  ' + gls_hyp6[item] + "   " + str(wer_list(hypotheses=[gls_hyp6[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Bone_motion (CNN) :  ' + gls_hyp7[item] + "   " + str(wer_list(hypotheses=[gls_hyp7[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('Bone_motion (LSTM):  ' + gls_hyp8[item] + "   " + str(wer_list(hypotheses=[gls_hyp8[item]], references=[gls_ref[item]])['wer']) + "\n")
                file.write('CONSENSUS:  ' + gls_hyp_consensus[item] + "   " + str(wer_list(hypotheses=[gls_hyp_consensus[item]], references=[gls_ref[item]])['wer']) + "\n\n")

    recoder.print_log('\tEpoch: {} {} done. Keypoint (CNN) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Keypoint (LSTM) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Bone (CNN) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con2['wer'], wer_results_con2['ins'], wer_results_con2['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Bone (LSTM) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results2['wer'], wer_results2['ins'], wer_results2['del']), f"{work_dir}/{mode}.txt")
    
    recoder.print_log('\tEpoch: {} {} done. Keypoint_motion (CNN) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con3['wer'], wer_results_con3['ins'], wer_results_con3['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Keypoint_motion (LSTM) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results3['wer'], wer_results3['ins'], wer_results3['del']), f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Bone_motion (CNN) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con4['wer'], wer_results_con4['ins'], wer_results_con4['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Bone_motion (LSTM) wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results4['wer'], wer_results4['ins'], wer_results4['del']), f"{work_dir}/{mode}.txt")
    
    recoder.print_log('-'*60, f"{work_dir}/{mode}.txt")
    recoder.print_log('>>> Ensemble Methods Results <<<', f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. Consensus Selection wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_consensus['wer'], wer_results_consensus['ins'], wer_results_consensus['del']),
        f"{work_dir}/{mode}.txt")
    
    gc.collect()
    
    best_result = wer_results_consensus
    return {"wer": best_result['wer'], "ins": best_result['ins'], 'del': best_result['del']}
 
