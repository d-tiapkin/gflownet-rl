import torch
from gfn.modules import GFNModule
from torchtyping import TensorType as TT
import torch.nn.functional as F

from gfn.containers import Trajectories, Transitions
from gfn.gflownet import GFlowNet
from gfn.states import States
from gfn.samplers import Sampler


class SACGFlowNet(GFlowNet):
    def __init__(
        self,
        actor: GFNModule,
        q1: GFNModule,
        q2: GFNModule,
        q1_target: GFNModule,
        q2_target: GFNModule,
        pb: GFNModule,
        on_policy: bool = False,
        entropy_coeff: float = 1.,
    ):
        super().__init__()
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.pb = pb
        self.on_policy = on_policy

        self.q1_target = q1_target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = q2_target
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.entropy_coeff = entropy_coeff

    def sample_trajectories(self, n_samples: int = 1000) -> Trajectories:
        sampler = Sampler(estimator=self.actor)
        trajectories = sampler.sample_trajectories(n_trajectories=n_samples)
        return trajectories

    def update_q_target(self, tau=0.05):
        if tau == 1.:
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            return
        with torch.no_grad():
            for param, target_param in zip(
                self.q1.parameters(), self.q1_target.parameters()
            ):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau,
                          out=target_param.data)
            for param, target_param in zip(
                self.q2.parameters(), self.q2_target.parameters()
            ):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau,
                          out=target_param.data)

    def compute_q_target(self, states: States):
        return torch.minimum(
            self.q1_target(states),
            self.q2_target(states)
        )

    def compute_v_target(self, states: States):
        with torch.no_grad():
            q_target = self.compute_q_target(states)
        logits = self.actor(states)
        logits[~states.forward_masks] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        log_policy = torch.log(probs + 1e-9)

        # To avoid 0 * inf
        logits[~states.forward_masks] = 0.0
        q_target[~states.forward_masks] = 0.0
        log_policy[~states.forward_masks] = 0.0
        return torch.sum(
            probs * (q_target - self.entropy_coeff * log_policy),
            dim=-1
        ).squeeze(-1)

    def get_td_preds_target(
        self, transitions: Transitions
    ):
        """
        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        q1_s = self.q1(states)
        q1_s[~states.forward_masks] = -float("inf")
        preds1 = torch.gather(q1_s, 1, actions.tensor).squeeze(-1)

        q2_s = self.q2(states)
        q2_s[~states.forward_masks] = -float("inf")
        preds2 = torch.gather(q2_s, 1, actions.tensor).squeeze(-1)

        targets = torch.zeros_like(preds1)

        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[~actions.is_exit]

        module_output = self.pb(valid_next_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        with torch.no_grad():
            valid_v_target_next = self.compute_v_target(valid_next_states)

        targets[~valid_transitions_is_done] = valid_log_pb_actions + valid_v_target_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]

        return preds1, preds2, targets

    def policy_loss(
        self, transitions: Transitions
    ):
        """Given a batch of transitions, calculate the loss for policy network.

        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        return -self.compute_v_target(states).mean()

    def q_loss(self, transitions: Transitions, loss_fn) -> TT[0, float]:
        preds1, preds2, targets = self.get_td_preds_target(transitions)
        loss = loss_fn(preds1, targets, reduction='mean')
        loss += loss_fn(preds2, targets, reduction='mean')

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
