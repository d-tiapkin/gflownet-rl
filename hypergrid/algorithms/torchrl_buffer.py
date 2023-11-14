from __future__ import annotations

from gfn.containers import Transitions
from gfn.env import Env

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class TorchRLReplayBuffer:
    """A replay buffer of transitions.

    Attributes:
        env: the Environment instance.
        capacity: the size of the buffer.
        training_objects: the buffer of objects used for training.
        terminating_states: a States class representation of $s_f$.
        objects_type: the type of buffer (transitions, trajectories, or states).
    """

    def __init__(
        self,
        env: Env,
        replay_buffer_size: int = 1000,
        prioritized: bool = True,
        alpha: float = 0.7,
        beta: float = 0.4,
        batch_size: int = 256
    ):
        """Instantiates a replay buffer.
        Args:
            env: the Environment instance.
            replay_buffer_size: size of the buffer
            prioritized: is buffer prioritized or not
            alpha:
            beta:
            batch_size:
        """
        self.env = env
        self.prioritized = prioritized
        if prioritized:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(replay_buffer_size),
                sampler=PrioritizedSampler(
                    max_capacity=replay_buffer_size,
                    alpha=alpha,
                    beta=beta,
                ),
                batch_size=batch_size,
                priority_key="td_error"
            )
            self.initial_beta = beta
        else:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(replay_buffer_size),
                batch_size=batch_size
            )

    def sample(self, batch_size : int | None = None):
        from_rb = self.replay_buffer.sample(batch_size)
        training_objects = tensor_dict2t(self.env, from_rb)
        return training_objects, from_rb

    def add(self, transitions : Transitions, td_error=None):
        self.replay_buffer.extend(t2tensor_dict(transitions, td_error))

    def update_priority(self, rb_batch : TensorDict, priorities):
        rb_batch["td_error"] = priorities
        self.replay_buffer.update_tensordict_priority(rb_batch)

    def update_beta(self, progress: float) -> None:
        add_beta = (1. - self.initial_beta) * progress
        self.replay_buffer._sampler._beta = self.initial_beta + add_beta


def t2tensor_dict(transitions: Transitions, td_error=None) -> TensorDict:
    # For simplicity we consider only forward transitions
    assert transitions.is_backward is False

    batch_shape = transitions.states.batch_shape

    states = transitions.states.tensor
    forward_masks = transitions.states.forward_masks

    actions = transitions.actions.tensor

    is_done = transitions.is_done

    next_states = transitions.next_states.tensor
    next_forward_masks = transitions.next_states.forward_masks

    result = TensorDict(
        {
            # Data of state
            "states": states,
            "forward_masks": forward_masks,
            # Data of actions
            "actions": actions,
            # Data of don
            "is_done": is_done,
            # Data of next state
            "next_states": next_states,
            "next_forward_masks": next_forward_masks,
        },
        batch_size=batch_shape
    )

    if td_error is not None:
        result.update({"td_error" : td_error})

    return result


def tensor_dict2t(env : Env, transitions_dict: TensorDict) -> Transitions:
    states = env.States(transitions_dict["states"])
    states.forward_masks = transitions_dict["forward_masks"]

    actions = env.Actions(transitions_dict["actions"])

    is_done = transitions_dict["is_done"]

    next_states = env.States(transitions_dict["next_states"])
    next_states.forward_masks = transitions_dict["next_forward_masks"]

    return Transitions(
        env=env,
        states=states,
        actions=actions,
        is_done=is_done,
        next_states=next_states
    )
