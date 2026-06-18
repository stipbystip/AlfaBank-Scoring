from dataclasses import dataclass
from pathlib import Path

AGG_DICT = {
    "pre_loans_credit_limit": ["mean", "max", "sum"],
    "pre_loans_next_pay_summ": ["mean", "max", "sum"],
    "pre_loans_outstanding": ["mean", "max", "sum"],
    "pre_loans_total_overdue": ["mean", "max", "sum"],
    "pre_loans_max_overdue_sum": ["mean", "max", "sum"],
    "pre_loans_credit_cost_rate": ["mean", "max"],
    "pre_loans5": ["mean", "sum"],
    "pre_loans530": ["mean", "sum"],
    "pre_loans3060": ["mean", "sum"],
    "pre_loans6090": ["mean", "sum"],
    "pre_loans90": ["mean", "sum"],
    "pre_util": ["mean", "max"],
    "pre_over2limit": ["mean", "max"],
    "pre_maxover2limit": ["mean", "max"],
    **{f"enc_paym_{i}": ["mean", "max", "sum"] for i in range(25)},
}

CAT_COLS = [
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "is_zero_loans5",
    "is_zero_loans530",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
    "is_zero_util",
    "is_zero_over2limit",
    "is_zero_maxover2limit",
    "pclose_flag",
    "fclose_flag",
]

SEQUENCE_MAX_LENGTH = 55

FEATURE_CARDINALITIES = {
    "pre_since_opened": 20,
    "pre_since_confirmed": 18,
    "pre_pterm": 18,
    "pre_fterm": 17,
    "pre_till_pclose": 17,
    "pre_till_fclose": 17,
    "pre_loans_credit_limit": 20,
    "pre_loans_next_pay_summ": 8,
    "pre_loans_outstanding": 10,
    "pre_loans_total_overdue": 2,
    "pre_loans_max_overdue_sum": 4,
    "pre_loans_credit_cost_rate": 15,
    "pre_loans5": 17,
    "pre_loans530": 20,
    "pre_loans3060": 10,
    "pre_loans6090": 6,
    "pre_loans90": 20,
    "is_zero_loans5": 2,
    "is_zero_loans530": 2,
    "is_zero_loans3060": 2,
    "is_zero_loans6090": 2,
    "is_zero_loans90": 2,
    "pre_util": 20,
    "pre_over2limit": 20,
    "pre_maxover2limit": 20,
    "is_zero_util": 2,
    "is_zero_over2limit": 2,
    "is_zero_maxover2limit": 2,
    "pclose_flag": 2,
    "fclose_flag": 2,
    "enc_loans_account_holder_type": 7,
    "enc_loans_credit_status": 7,
    "enc_loans_credit_type": 8,
    "enc_loans_account_cur": 4,
}

for i in range(25):
    FEATURE_CARDINALITIES[f"enc_paym_{i}"] = 5

FEATURE_COLS = list(FEATURE_CARDINALITIES.keys())


@dataclass(frozen=True)
class DataPaths:
    base_data_dir: Path
    train_data: Path
    test_data: Path
    train_target: Path
    sample_submission: Path | None = None
    pseudo_target: Path | None = None

    @classmethod
    def from_base_dir(
        cls,
        base_data_dir: str | Path,
        pseudo_target: str | Path | None = None,
    ) -> "DataPaths":
        base = Path(base_data_dir)
        return cls(
            base_data_dir=base,
            train_data=base / "train_data.parquet",
            test_data=base / "test_data.parquet",
            train_target=base / "train_target.csv",
            sample_submission=base / "sample_submission.csv",
            pseudo_target=Path(pseudo_target) if pseudo_target else None,
        )


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int = 5
    random_state: int = 42
    catboost_iterations: int = 5000
    catboost_depth: int = 8
    catboost_learning_rate: float = 0.02
    transformer_epochs: int = 15
    mlp_epochs: int = 40
    batch_size_sequences: int = 512
    batch_size_embeddings: int = 2048
    device: str | None = None
