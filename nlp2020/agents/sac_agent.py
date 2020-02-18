import os
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from collections import deque
from torch.utils.data.sampler import WeightedRandomSampler



def to_batch(state, action, reward, next_state, done, device):
    state = torch.ByteTensor(
        state).unsqueeze(0).to(device).float() / 255.
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.ByteTensor(
        next_state).unsqueeze(0).to(device).float() / 255.
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class MultiStepBuff:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
            }

    def append(self, state, action, reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class DummyMemory(dict):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, action_shape, device):
        super(DummyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.reset()

    def reset(self):
        for key in self.state_keys:
            self[key] = []

        self['action'] = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i] = self['state'][_index]
            next_states[i] = self['next_state'][_index]

        states = torch.ByteTensor(states).to(self.device).float() 
        next_states = torch.ByteTensor(next_states).to(self.device).float() 
        
        actions = torch.FloatTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self['state'])

    def get(self):
        return dict(self)

    def load(self, memory):
        for key in self.state_keys:
            self[key].extend(memory[key])

        num_data = len(memory['state'])
        if self._p + num_data <= self.capacity:
            for key in self.np_keys:
                self[key][self._p:self._p+num_data] = memory[key]
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            for key in self.np_keys:
                self[key][self._p:] = memory[key][:mid_index]
                self[key][:end_index] = memory[key][mid_index:]

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity
        self.truncate()
        assert self._n == len(self)


class DummyMultiStepMemory(DummyMemory):

    def __init__(self, capacity, state_shape, action_shape, device,
                 gamma=0.99, multi_step=3):
        super(DummyMultiStepMemory, self).__init__(
            capacity, state_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done)


class DummyPrioritizedMemory(DummyMultiStepMemory):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done', 'priority']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, action_shape, device, gamma=0.99,
                 multi_step=3, alpha=0.6, beta=0.4, beta_annealing=0.001,
                 epsilon=1e-4):
        super(DummyPrioritizedMemory, self).__init__(
            capacity, state_shape, action_shape, device, gamma, multi_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def reset(self):
        super(DummyPrioritizedMemory, self).reset()
        self['priority'] = np.empty((self.capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, next_state, done, error,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self['priority'][self._p] = self.calc_priority(error)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self['priority'][self._p] = self.calc_priority(error)
            self._append(
                state, action, reward, next_state, done)

    def update_priority(self, indices, errors):
        self['priority'][indices] = np.reshape(
            self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def get(self):
        state_dict = {key: self[key] for key in self.state_keys}
        np_dict = {key: self[key][:self._n] for key in self.np_keys}
        state_dict.update(**np_dict)
        return state_dict

    def sample(self, batch_size):
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(
            self['priority'][:self._n, 0], batch_size)
        indices = list(sampler)

        batch = self._sample(indices, batch_size)
        priorities = np.array(self['priority'][indices], dtype=np.float32)
        priorities = priorities / np.sum(self['priority'][:self._n])

        weights = (self._n * priorities) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(
            weights).view(batch_size, -1).to(self.device)

        return batch, indices, weights


def weights_init_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def create_nn(obs_dim, num_actions):
    return nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_actions),
    ).apply(weights_init_he)  # The author of the paper used He's initializer.


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, obs_dim, num_actions):
        super(TwinnedQNetwork, self).__init__()
        self.Q1 = create_nn(obs_dim, num_actions)
        self.Q2 = create_nn(obs_dim, num_actions)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, obs_dim, num_actions):
        super(CateoricalPolicy, self).__init__()
        self.policy = create_nn(obs_dim, num_actions)

    def act(self, states):
        # act with greedy policy
        action_logits = self.policy(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        # act with exploratory policy
        action_probs = F.softmax(self.policy(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs




class SacDiscreteAgent:

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 target_entropy_ratio=0.98, lr=0.0003, memory_size=1000000,
                 gamma=0.99, target_update_type='soft',
                 target_update_interval=8000, tau=0.005, multi_step=1,
                 per=False, alpha=0.6, beta=0.4, beta_annealing=0.0001,
                 grad_clip=5.0, update_every_n_steps=4,
                 learnings_per_update=1, start_steps=1000, log_interval=10,
                 eval_interval=1000, cuda=True, seed=0):
        self.env = env
        self.test_env = test_env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.policy = CateoricalPolicy(
            self.env.observation_space.n, self.env.action_space.n
            ).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.n, self.env.action_space.n
            ).to(device=self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.n, self.env.action_space.n
            ).to(device=self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=1e-4)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr, eps=1e-4)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr, eps=1e-4)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy =\
            -np.log(1.0/self.env.action_space.n) * target_entropy_ratio
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr, eps=1e-4)

        # DummyMemory efficiently stores FrameStacked states.
        if per:
            # replay memory with prioritied experience replay
            self.memory = DummyPrioritizedMemory(
                memory_size, self.env.observation_space.n,
                (1,), self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            self.memory = DummyMultiStepMemory(
                memory_size, self.env.observation_space.n,
                (1,), self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.target_update_type = target_update_type
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.per = per
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.grad_clip = grad_clip
        self.update_every_n_steps = update_every_n_steps
        self.learnings_per_update = learnings_per_update
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_every_n_steps == 0\
            and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # act without randomness
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs =\
                self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).mean(dim=1, keepdim=True)

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            # clip reward to [-1.0, 1.0]
            clipped_reward = max(min(reward, 1.0), -1.0)

            if self.per:
                batch = to_batch(
                    state, action, clipped_reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, _ = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done,
                    error, episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.learnings_per_update):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)
    
            print(f'episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1
        if self.target_update_type == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.learning_steps % self.target_update_interval == 0:
            hard_update(self.critic_target, self.critic)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)
        update_params(self.alpha_optim, None, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (log of) probabilities to calculate expectations of Q and entropies
        _, action_probs, log_action_probs = self.policy.sample(states)
        # Q for every actions to calculate expectations of Q
        q1, q2 = self.critic(states)
        q = torch.min(q1, q2)

        # expectations of entropies
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)
        # expectations of Q
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.test_env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f} +/- {std_return:<5.1f}')
        print('-' * 60)

    def save_models(self):
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()