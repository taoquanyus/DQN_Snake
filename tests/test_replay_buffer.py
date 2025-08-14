import numpy as np

from replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample():
    buffer = ReplayBuffer(5)
    state = np.array([1, 2, 3], dtype=np.float32)
    next_state = np.array([1, 2, 4], dtype=np.float32)
    buffer.add(state, 0, 1, next_state, False)

    assert buffer.size() == 1

    states, actions, rewards, next_states, dones = buffer.sample(1)

    np.testing.assert_array_equal(states[0], state)
    assert actions[0] == 0
    assert rewards[0] == 1
    np.testing.assert_array_equal(next_states[0], next_state)
    assert dones[0] is False


def test_replay_buffer_capacity():
    buffer = ReplayBuffer(2)
    buffer.add(1, 1, 1, 1, False)
    buffer.add(2, 2, 2, 2, False)
    buffer.add(3, 3, 3, 3, False)

    assert buffer.size() == 2

    states, *_ = buffer.sample(2)
    assert set(states.tolist()) == {2, 3}
