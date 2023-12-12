from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class TrainingConfig:
    base_model_id: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    device: str = "cpu"  # {cpu, cuda, mps}
    test_size: float = 0.3
    model_path: str = "ml_models/model_v1"
    model_parameters: Dict[str, Union[int, str]] = field(default_factory=lambda: {
        'metric': "f1",
        'batch_size': 32,
        'num_iterations': 20,
        'num_epochs': 1
    })
    data_path: str = "DS test_data.csv"
    text_column: str = "Messages"
    label_column: str = "gen_label"

training_config = TrainingConfig()
