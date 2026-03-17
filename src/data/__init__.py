"""
Data loading, preprocessing, epoching, and splitting.
"""

from src.data.dataset import EEGDataset
from src.data.loading import download_dataset, load_raw_subjects
from src.data.preprocessing import epoch_subjects, epoch_with_params
from src.data.splits import make_dataloaders, subject_split

__all__ = [
    "EEGDataset",
    "download_dataset",
    "load_raw_subjects",
    "epoch_subjects",
    "epoch_with_params",
    "subject_split",
    "make_dataloaders",
]