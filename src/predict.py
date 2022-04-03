
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

from .data.dataset import PCASegmentationDataset
from torch.utils.data._utils.collate import default_collate
from .metrics import Metrics

import numpy as np

@dataclass 
class SlicePrediction:
    mri: np.ndarray
    pred_mask: np.ndarray
    true_mask: np.ndarray


@dataclass
class SegmentationMetrics:
    dice: float 
    loss: float 
    jaccard: float
    
    
@dataclass
class CaseReport:
    metrics: SegmentationMetrics
    slice_predictions: List[SlicePrediction]


@dataclass
class Report:
    overall_metrics: SegmentationMetrics
    case_reports: Dict[int, CaseReport]


def predict(model: torch.nn.Module, dataset: PCASegmentationDataset, device: Optional[str] = None):
    
    assert not dataset.transform.use_augmentations, f"cannot predict on augmented dataset."
    
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def patient_batch_generator():
        for case_num in dataset.get_case_list():
            case_indices = dataset.get_indices_for_case_number(case_num)
            batch = []
            for idx in case_indices: 
                mri, seg = dataset[idx]
                with dataset.raw():
                    raw_mri, _ = dataset[idx]
                batch.append((mri, seg, raw_mri))
        
            batch = default_collate(batch)
            
            yield case_num, batch
        
    metrics = Metrics()
    case_reports = {}
    
    for case_num, batch in patient_batch_generator():
        
        mri, seg, raw_mri = batch
        mri = mri.to(device)
        seg = seg.to(device)
        raw_mri = raw_mri.to(device)
        
        with torch.no_grad():
            logits = model(mri)
            case_metrics = SegmentationMetrics(**metrics(logits, seg))        
            pred_mask = torch.argmax(logits, dim=1)
            
        raw_mri = raw_mri.cpu().detach().numpy()
        pred_mask = pred_mask.cpu().detach().numpy()
        mask = seg.cpu().detach().numpy()
        
        slice_predictions = [SlicePrediction(
            mri=mri, 
            pred_mask=pred, 
            true_mask=true
        ) for mri, pred, true in zip(raw_mri, pred_mask, mask)]
        
        case_report = CaseReport(case_metrics, slice_predictions)
        case_reports[case_num] = case_report
        
    overall_metrics = SegmentationMetrics(**metrics.compute())
    
    return Report(overall_metrics, case_reports)