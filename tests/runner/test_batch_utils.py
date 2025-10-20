from hyperlax.hyperparam.base_types import HyperparamBatchGroup
from hyperlax.runner.batch_utils import group_samples_into_batches, slice_batches


# --- Test slice_batches ---
def create_dummy_group(num_batches: int) -> HyperparamBatchGroup:
    """Helper to create a dummy HyperparamBatchGroup."""
    return HyperparamBatchGroup(
        non_vec_values={"a": 1},
        vec_batches=[{"b": i} for i in range(num_batches)],
        sample_ids=list(range(num_batches)),
        default_values={"c": 2},
    )


def test_slice_batches_no_slicing_needed():
    group = create_dummy_group(3)
    sliced = slice_batches([group], max_batch_size=4, min_batch_size=2)
    assert len(sliced) == 1
    assert len(sliced[0].sample_ids) == 3


def test_slice_batches_exact_division():
    group = create_dummy_group(6)
    sliced = slice_batches([group], max_batch_size=3, min_batch_size=2)
    assert len(sliced) == 2
    assert [len(g.sample_ids) for g in sliced] == [3, 3]
    assert sliced[0].sample_ids == [0, 1, 2]
    assert sliced[1].sample_ids == [3, 4, 5]


def test_slice_batches_with_remainder_above_min():
    group = create_dummy_group(5)
    sliced = slice_batches([group], max_batch_size=3, min_batch_size=2)
    assert len(sliced) == 2
    assert [len(g.sample_ids) for g in sliced] == [3, 2]


def test_slice_batches_with_remainder_below_min_redistributes():
    # Total 7, max 3, min 2.
    # Num slices = ceil(7/3) = 3. base = 7//3=2, rem=1. Sizes: [2+1, 2, 2] -> [3, 2, 2]. All >= min.
    group = create_dummy_group(7)
    sliced = slice_batches([group], max_batch_size=3, min_batch_size=2)
    assert len(sliced) == 3
    assert [len(g.sample_ids) for g in sliced] == [3, 2, 2]
    assert sliced[0].sample_ids == [0, 1, 2]
    assert sliced[1].sample_ids == [3, 4]
    assert sliced[2].sample_ids == [5, 6]


def test_slice_batches_with_tiny_remainder_complex_redistribution():
    # Total 10, max 4, min 3.
    # Num slices = ceil(10/4) = 3. base=10//3=3, rem=1. Sizes: [3+1, 3, 3] -> [4, 3, 3]. All >= min.
    group = create_dummy_group(10)
    sliced = slice_batches([group], max_batch_size=4, min_batch_size=3)
    assert len(sliced) == 3
    assert [len(g.sample_ids) for g in sliced] == [4, 3, 3]


def test_slice_batches_multiple_groups():
    group1 = create_dummy_group(5)
    group2 = create_dummy_group(2)
    sliced = slice_batches([group1, group2], max_batch_size=2, min_batch_size=1)
    assert len(sliced) == 4
    # group1 (5) -> [2, 2, 1]
    # group2 (2) -> [2]
    assert [len(g.sample_ids) for g in sliced] == [2, 2, 1, 2]
    assert sliced[0].non_vec_values == {"a": 1}
    assert sliced[3].non_vec_values == {"a": 1}


# --- Test group_samples_into_batches ---
def test_group_samples_into_batches_single_group():
    samples = {
        "non_vec1": [10, 10, 10],
        "non_vec2": ["a", "a", "a"],
        "vec1": [1.1, 2.2, 3.3],
        "vec2": [True, False, True],
        "some.sample_id": [0, 1, 2],
    }
    vectorized_keys = {"vec1", "vec2", "some.sample_id"}
    non_vectorized_keys = {"non_vec1", "non_vec2"}
    defaults = {"default1": 99}

    groups = group_samples_into_batches(samples, vectorized_keys, non_vectorized_keys, defaults, "some.sample_id")

    assert len(groups) == 1
    group = groups[0]
    assert group.non_vec_values == {"non_vec1": 10, "non_vec2": "a"}
    assert group.default_values == {"default1": 99}
    assert group.sample_ids == [0, 1, 2]
    assert len(group.vec_batches) == 3
    assert group.vec_batches[0] == {"vec1": 1.1, "vec2": True}
    assert group.vec_batches[2] == {"vec1": 3.3, "vec2": True}


def test_group_samples_into_batches_multiple_groups():
    samples = {"non_vec1": [10, 20, 10, 20], "vec1": [1.1, 2.2, 3.3, 4.4], "some.sample_id": [0, 1, 2, 3]}
    vectorized_keys = {"vec1", "some.sample_id"}
    non_vectorized_keys = {"non_vec1"}
    defaults = {}

    groups = group_samples_into_batches(samples, vectorized_keys, non_vectorized_keys, defaults, "some.sample_id")

    assert len(groups) == 2

    # Sort groups by sample_id to make test deterministic
    groups.sort(key=lambda g: g.sample_ids[0])

    group1 = groups[0]  # Should be the one for non_vec1 = 10
    assert group1.non_vec_values == {"non_vec1": 10}
    assert group1.sample_ids == [0, 2]
    assert len(group1.vec_batches) == 2
    assert group1.vec_batches[0] == {"vec1": 1.1}
    assert group1.vec_batches[1] == {"vec1": 3.3}

    group2 = groups[1]  # Should be the one for non_vec1 = 20
    assert group2.non_vec_values == {"non_vec1": 20}
    assert group2.sample_ids == [1, 3]
    assert len(group2.vec_batches) == 2
    assert group2.vec_batches[0] == {"vec1": 2.2}
    assert group2.vec_batches[1] == {"vec1": 4.4}


def test_group_samples_no_non_vectorized():
    samples = {"vec1": [1.1, 2.2, 3.3], "vec2": [True, False, True], "some.sample_id": [0, 1, 2]}
    vectorized_keys = {"vec1", "vec2", "some.sample_id"}
    non_vectorized_keys = set()
    defaults = {}

    groups = group_samples_into_batches(samples, vectorized_keys, non_vectorized_keys, defaults, "some.sample_id")

    assert len(groups) == 1
    group = groups[0]
    assert group.non_vec_values == {}
    assert group.sample_ids == [0, 1, 2]
    assert len(group.vec_batches) == 3
