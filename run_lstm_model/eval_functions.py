
from torchmetrics import PrecisionRecallCurve, AUROC
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import argparse
from run_lstm_model import get_model
from data_module import ClickstreamDataModule
from npz_extractor import NPZExtractor


from torchmetrics import PrecisionRecallCurve, AUROC
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
def _extract_features(tensor, n_sequences):
        

        features, shifters = (
             tensor[:, 5:5+17], tensor[:, 5+17:]
        )

        shifters = torch.reshape(shifters, (shifters.shape[0], n_sequences, 18))
        shifters = shifters[:, :, 1:]
        features = torch.flip(torch.cat((features.unsqueeze(1), shifters), dim=1), dims=[1])
        return features

  
def auc_by_user_bin(model, dataset, n_sequences, dataset_type):
    bin_container = {}
    model = model.to('cuda') if torch.cuda.is_available() else model
    model.eval()
    model.half()
    bin_list = [0, 25, 50, 75, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    bin_container = {bin: [] for bin in bin_list if bin != 0}
    bin_container["max"] = []
    print(f'Running AUC by user bin for dataset_val: {dataset_type}')
    """
    Wrap dataloader in TQDM to get progress bar
    """
    # pr_curve = PrecisionRecallCurve(task='binary')
    for i, bin in enumerate(bin_list):
        if i == 0:
             continue
        pr_curve = PrecisionRecallCurve(task='binary')
        dataset_pbar = tqdm(dataset)
        dataset_pbar.set_description(f'Predictions: {bin_list[i-1]} {bin}')
        for indx, batch in enumerate(dataset_pbar):
            batch = batch.to('cuda') if torch.cuda.is_available() else batch
            
            batch_min = batch[(batch[:, 1] > bin_list[i-1]) & (batch[:, 1] <= bin)]
            labels_min = batch_min[:, 0].int().unsqueeze(1)
            features_min = _extract_features(batch_min, n_sequences)
            with torch.no_grad():
                if features_min.shape[0] > 0:
                    features_min = features_min.half()
                    preds_min = model(features_min) 
                    preds_min = nn.Sigmoid()(preds_min)
                    prec_min, rec_min, thresh = pr_curve(preds_min, labels_min)
                    bin_container[bin].append({
                        'precision': prec_min.detach().cpu().numpy().tolist(),
                        'recall': rec_min.detach().cpu().numpy().tolist(),
                        'threshold': thresh.detach().cpu().numpy().tolist()
                    })
            
                if bin == max(bin_list):
                    batch_max = batch[(batch[:, 1] > bin)]
                    labels_max = batch_max[:, 0].int().unsqueeze(1)
                    features_max = _extract_features(batch_max, n_sequences)
                    if features_max.shape[0] > 0:
                        features_max = features_max.half()
                        preds_max = model(features_max)
                        preds_max = nn.Sigmoid()(preds_max)
                        prec_max, rec_max, thresh = pr_curve(preds_max, labels_max)
                        bin_container["max"].append({
                            'precision': prec_max.detach().cpu().numpy().tolist(),
                            'recall': rec_max.detach().cpu().numpy().tolist(),
                            'threshold': thresh.detach().cpu().numpy().tolist()
                        })

    return bin_container

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--n_files', type=int, default=61)
    parser.add_argument('--partition', type=str, default=None)


def main():


    for seq in [10, 20, 30, 40]:
        model = get_model(
            'ordinal',
            20,
            seq,
            32,
            .2,
            0.001,
            8192,
            True,
            61
        )

        model_heuristic = get_model(
            'ordinal',
            20,
            seq,
            32,
            .2,
            0.001,
            8192,
            False,
            61
        )

        npz_extractor = NPZExtractor(
            'torch_ready_data_4',
            61,
            seq+1,
            20
        )

        data_module = ClickstreamDataModule(npz_extractor.get_dataset_pointer(), 8192, seq+1, 20)

        val_data, test_data = data_module.val_dataloader(), data_module.test_dataloader()

        """
        set checkpoint path dict for each model and heuristic"
        """
        model.load_from_checkpoint(CHECKPOINT_MATRIX[str(seq)])
        model_heuristic.load_from_checkpoint(CHECKPOINT_MATRIX[str(seq) + '_heuristic'])

        print(f'Running AUC by user bin for model: {CHECKPOINT_MATRIX[str(seq)]}')
        results_val, results_test = auc_by_user_bin(model, val_data, seq+1, 'validation'), auc_by_user_bin(model, test_data, seq+1, 'test')
        
        print(f'Running AUC by user bin for model: {CHECKPOINT_MATRIX[str(seq) + "_heuristic"]}')
        results_val_heuristic, results_test_heuristic = auc_by_user_bin(model_heuristic, val_data, seq+1, 'validation'), auc_by_user_bin(model_heuristic, test_data, seq+1, 'test')

        with open(f'auc_by_user_bin_{seq}_val.json', 'w') as f:
            json.dump(results_val, f)

        with open(f'auc_by_user_bin_{seq}_test.json', 'w') as f:
            json.dump(results_test, f)

        
        with open(f'auc_by_user_bin_{seq}_val_heuristic.json', 'w') as f:
            json.dump(results_val_heuristic, f)
        
        with open(f'auc_by_user_bin_{seq}_test_heuristic.json', 'w') as f:
            json.dump(results_test_heuristic, f)



     
