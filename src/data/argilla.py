from datasets import load_dataset
from src.config.settings import settings

train_dataset = load_dataset(path=settings.hf_dataset_path,split=settings.train_spilt)
test_dataset = load_dataset(path=settings.hf_dataset_path,split=settings.test_split)

# print(train_dataset.features)