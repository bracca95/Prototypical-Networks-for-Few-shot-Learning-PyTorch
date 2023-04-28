from typing import TypeVar, Optional, List


class Consts:
    T = TypeVar("T")

    DATASET_MEAN: Optional[List[float]] = [0.5366]
    DATASET_STD: Optional[List[float]] = [0.0619]