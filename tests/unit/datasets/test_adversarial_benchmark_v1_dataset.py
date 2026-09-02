# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import Counter

from pyrit.datasets import SeedDatasetProvider
from pyrit.models import SeedObjective

DATASET_NAME = "adversarial_benchmark_v1"
EXPECTED_CATEGORY_COUNTS = {
    "election_critical_information": 5,
    "hate_v3": 5,
    "inference_sensitive_attributes": 5,
    "offensive_cyber_v2": 5,
    "self_harm_v3": 5,
    "sensitive_data_leakage": 5,
    "sexual_v3": 5,
    "violence_v3": 5,
}


async def test_adversarial_benchmark_v1_resolves_by_name_async() -> None:
    datasets = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[DATASET_NAME])

    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset.dataset_name == DATASET_NAME
    assert len(dataset.seeds) == 40
    assert all(isinstance(seed, SeedObjective) for seed in dataset.seeds)
    assert all(seed.dataset_name == DATASET_NAME for seed in dataset.seeds)
    assert len(dataset.seed_groups) == 40
    assert all(not group.prompts for group in dataset.seed_groups)

    category_counts = Counter(category for seed in dataset.objectives for category in seed.harm_categories)
    assert category_counts == EXPECTED_CATEGORY_COUNTS
