import pytest

from replay_buffer import ReplayBuffer


def test_sample_raises_when_requesting_too_many_transitions():
    buffer = ReplayBuffer(3)
    buffer.add(1, 2, 3, 4, False)
    with pytest.raises(ValueError):
        buffer.sample(2)
