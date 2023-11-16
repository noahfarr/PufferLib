from pdb import set_trace as T
import numpy as np
import cv2

import os
import random
import time
import uuid

from collections import defaultdict
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.args
import pufferlib.utils
import pufferlib.emulation
import pufferlib.vectorization
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool


@pufferlib.dataclass
class Performance:
    total_uptime = 0
    total_updates = 0
    total_agent_steps = 0
    epoch_time = 0
    epoch_sps = 0
    evaluation_time = 0
    evaluation_sps = 0
    evaluation_memory = 0
    evaluation_pytorch_memory = 0
    env_time = 0
    env_sps = 0
    inference_time = 0
    inference_sps = 0
    train_time = 0
    train_sps = 0
    train_memory = 0
    train_pytorch_memory = 0

@pufferlib.dataclass
class Losses:
    policy_loss = 0
    value_loss = 0
    entropy = 0
    old_approx_kl = 0
    approx_kl = 0
    clipfrac = 0
    explained_variance = 0

@pufferlib.dataclass
class Charts:
    global_step = 0
    SPS = 0
    learning_rate = 0
    episodic_length = 0
    episodic_return = 0

def init(
        self: object = None,
        config: pufferlib.args.CleanPuffeRL = None,
        exp_name: str = None,
        track: bool = False,

        # Agent
        agent: nn.Module = None,
        agent_creator: callable = None,
        agent_kwargs: dict = None,

        # Environment
        env_creator: callable = None,
        env_creator_kwargs: dict = None,
        vectorization: ... = pufferlib.vectorization.Serial,

        # Policy Pool options
        policy_selector: callable = pufferlib.policy_pool.random_selector,
    ):
    if config is None:
        config = pufferlib.args.CleanPuffeRL()

    if exp_name is None:
        exp_name = str(uuid.uuid4())[:8]
    wandb = None
    if track:
        import wandb

    start_time = time.time()
    seed_everything(config.seed, config.torch_deterministic)
    total_updates = config.total_timesteps // config.batch_size
    envs_per_worker = config.num_envs // config.num_cores

    device = 'cuda' if config.cuda else 'cpu'
    obs_device = 'cpu' if config.cpu_offload else device
    assert config.num_cores * envs_per_worker == config.num_envs

    # Create environments, agent, and optimizer
    init_profiler = pufferlib.utils.Profiler(memory=True)
    with init_profiler:
        buffers = [
            vectorization(
                env_creator,
                env_kwargs=env_creator_kwargs,
                num_workers=config.num_cores,
                envs_per_worker=envs_per_worker,
            )
            for _ in range(config.num_buffers)
        ]

    obs_shape = buffers[0].single_observation_space.shape
    atn_shape = buffers[0].single_action_space.shape
    num_agents = buffers[0].num_agents
    total_agents = num_agents * config.num_envs

    # If data_dir is provided, load the resume state
    resume_state = {}
    path = os.path.join(config.data_dir, exp_name)
    if os.path.exists(path):
        trainer_path = os.path.join(path, 'trainer_state.pt')
        resume_state = torch.load(trainer_path)
        model_path = os.path.join(path, resume_state["model_name"])
        agent = torch.load(model_path, map_location=device)
        print(f'Resumed from update {resume_state["update"]} '
              f'with policy {resume_state["model_name"]}')
    else:
        agent = pufferlib.emulation.make_object(
            agent, agent_creator, buffers[:1], agent_kwargs)

    global_step = resume_state.get("global_step", 0)
    agent_step = resume_state.get("agent_step", 0)
    update = resume_state.get("update", 0)

    optimizer = optim.Adam(agent.parameters(),
        lr=config.learning_rate, eps=1e-5)
    opt_state = resume_state.get("optimizer_state_dict", None)
    if opt_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])


    # Create policy pool
    policy_pool = pufferlib.policy_pool.PolicyPool(
        agent, total_agents, atn_shape, device, path,
        config.pool_kernel, policy_selector,
    )

    # Allocate Storage
    storage_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
    next_lstm_state = []
    for i, envs in enumerate(buffers):
        envs.async_reset(config.seed + i)
        if not hasattr(agent, 'lstm'):
            next_lstm_state.append(None)
        else:
            shape = (agent.lstm.num_layers, total_agents, agent.lstm.hidden_size)
            next_lstm_state.append((
                torch.zeros(shape).to(device),
                torch.zeros(shape).to(device),
            ))
    obs=torch.zeros(config.batch_size + 1, *obs_shape).to(obs_device)
    actions=torch.zeros(config.batch_size + 1, *atn_shape, dtype=int).to(device)
    logprobs=torch.zeros(config.batch_size + 1).to(device)
    rewards=torch.zeros(config.batch_size + 1).to(device)
    dones=torch.zeros(config.batch_size + 1).to(device)
    truncateds=torch.zeros(config.batch_size + 1).to(device)
    values=torch.zeros(config.batch_size + 1).to(device)
    storage_profiler.stop()

    # Original CleanRL charts for comparison
    charts = Charts(
        global_step=global_step,
        learning_rate=config.learning_rate,
    )

    #"charts/actions": wandb.Histogram(b_actions.cpu().numpy()),
    init_performance = pufferlib.namespace(
        init_time = time.time() - start_time,
        init_env_time = init_profiler.elapsed,
        init_env_memory = init_profiler.memory,
        tensor_memory = storage_profiler.memory,
        tensor_pytorch_memory = storage_profiler.pytorch_memory,
    )
 
    return pufferlib.namespace(self,
        # Agent, Optimizer, and Environment
        config=config,
        agent = agent,
        optimizer = optimizer,
        buffers = buffers,
        policy_pool = policy_pool,

        # Logging
        exp_name = exp_name,
        wandb = wandb,
        charts = charts,
        losses = Losses(),
        init_performance = init_performance,
        performance = Performance(),

        # Storage
        sort_keys = [],
        next_lstm_state = next_lstm_state,
        obs = obs,
        actions = actions,
        logprobs = logprobs,
        rewards = rewards,
        dones = dones,
        values = values,

        # Misc
        total_updates = total_updates,
        update = update,
        global_step = global_step,
        device = device,
        obs_device = obs_device,
        start_time = start_time,
        buf = 0,
    )

@pufferlib.utils.profile
def evaluate(data):
    config = data.config
    # TODO: Handle update on resume
    if data.wandb is not None and data.performance.total_uptime > 0:
        data.wandb.log({
            **{f'charts/{k}': v for k, v in data.charts.items()},
            **{f'losses/{k}': v for k, v in data.losses.items()},
            **{f'performance/{k}': v
                for k, v in data.performance.items()},
            **{f'stats/{k}': v for k, v in data.stats.items()},
            **{f'skillrank/{policy}': elo
                for policy, elo in data.policy_pool.ranker.ratings.items()},
            'global_step': data.charts.global_step,
            'agent_steps': data.charts.agent_steps,
        })

    data.policy_pool.update_policies()
    performance = defaultdict(list)
    env_profiler = pufferlib.utils.Profiler()
    inference_profiler = pufferlib.utils.Profiler()
    eval_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()

    ptr = step = padded_steps_collected = agent_steps_collected = 0
    infos = defaultdict(lambda: defaultdict(list))
    while True:
        buf = data.buf
        step += 1
        if ptr == config.batch_size + 1:
            break

        with env_profiler:
            o, r, d, t, i, mask = data.buffers[buf].recv()

        '''
        for profile in data.buffers[buf].profile():
            for k, v in profile.items():
                performance[k].append(v["delta"])
        '''

        i = data.policy_pool.update_scores(i, "return")
        o = torch.Tensor(o).to(data.obs_device)
        r = torch.Tensor(r).float().to(data.device).view(-1)
        d = torch.Tensor(d).float().to(data.device).view(-1)

        agent_steps_collected += sum(mask)
        padded_steps_collected += len(mask)
        with inference_profiler, torch.no_grad():
            actions, logprob, value, data.next_lstm_state[buf] = data.policy_pool.forwards(
                    o.to(data.device), data.next_lstm_state[buf])
            value = value.flatten()

        # Index alive mask with policy pool idxs...
        # TODO: Find a way to avoid having to do this
        learner_mask = mask * data.policy_pool.mask

        for idx in np.where(learner_mask)[0]:
            if ptr == config.batch_size + 1:
                break
            data.obs[ptr] = o[idx]
            data.values[ptr] = value[idx]
            data.actions[ptr] = actions[idx]
            data.logprobs[ptr] = logprob[idx]
            data.sort_keys.append((buf, idx, step))
            if len(d) != 0:
                data.rewards[ptr] = r[idx]
                data.dones[ptr] = d[idx]
            ptr += 1


        for policy_name, policy_i in i.items():
            for agent_i in policy_i:
                for name, dat in unroll_nested_dict(agent_i):
                    infos[policy_name][name].append(dat)

        with env_profiler:
            data.buffers[buf].send(actions.cpu().numpy())
        data.buf = (data.buf + 1) % config.num_buffers

    data.global_step += config.batch_size
    eval_profiler.stop()

    charts = data.charts
    charts.reward = float(torch.mean(data.rewards))
    charts.agent_steps = data.global_step
    charts.SPS = int(padded_steps_collected / eval_profiler.elapsed)
    charts.global_step = data.global_step

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_agent_steps = data.global_step
    perf.env_time = env_profiler.elapsed
    perf.env_sps = int(agent_steps_collected / env_profiler.elapsed)
    perf.inference_time = inference_profiler.elapsed
    perf.inference_sps = int(padded_steps_collected / inference_profiler.elapsed)
    perf.eval_time = eval_profiler.elapsed
    perf.eval_sps = int(padded_steps_collected / eval_profiler.elapsed)
    perf.eval_memory = eval_profiler.end_mem
    perf.eval_pytorch_memory = eval_profiler.end_torch_mem

    data.stats = {}
    for k, v in infos['learner'].items():
        if 'Task_eval_fn' in k:
            # Temporary hack for NMMO competition
            continue
        if 'exploration_map' in k:
            import cv2
            bg = cv2.imread('full_map_v2a.png')
            overlay = make_pokemon_red_overlay(bg, sum(v))
            if data.wandb is not None:
                data.stats['Media/exploration_map'] = data.wandb.Image(overlay)
        try: # TODO: Better checks on log data types
            data.stats[k] = np.mean(v)
        except:
            continue

    if config.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    if done_training(data):
        raise RuntimeError(
            f"Max training updates {data.total_updates} already reached")

    config = data.config
    # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
    train_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True)
    train_profiler.start()

    if config.anneal_lr:
        frac = 1.0 - (data.update - 1.0) / data.total_updates
        lrnow = frac * config.learning_rate
        data.optimizer.param_groups[0]["lr"] = lrnow

    num_minibatches = config.batch_size // config.bptt_horizon // config.batch_rows
    idxs = sorted(range(len(data.sort_keys)), key=data.sort_keys.__getitem__)
    data.sort_keys = []
    b_idxs = (
        torch.Tensor(idxs)
        .long()[:-1]
        .reshape(config.batch_rows, num_minibatches, config.bptt_horizon)
        .transpose(0, 1)
    )

    # bootstrap value if not done
    with torch.no_grad():
        advantages = torch.zeros(config.batch_size, device=data.device)
        lastgaelam = 0
        for t in reversed(range(config.batch_size)):
            i, i_nxt = idxs[t], idxs[t + 1]
            nextnonterminal = 1.0 - data.dones[i_nxt]
            nextvalues = data.values[i_nxt]
            delta = (
                data.rewards[i_nxt]
                + config.gamma * nextvalues * nextnonterminal
                - data.values[i]
            )
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )

    # Flatten the batch
    data.b_obs = b_obs = data.obs[b_idxs]
    b_actions = data.actions[b_idxs]
    b_logprobs = data.logprobs[b_idxs]
    b_dones = data.dones[b_idxs]
    b_values = data.values[b_idxs]
    b_advantages = advantages.reshape(
        config.batch_rows, num_minibatches, config.bptt_horizon
    ).transpose(0, 1)
    b_returns = b_advantages + b_values

    # Optimizing the policy and value network
    train_time = time.time()
    pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(num_minibatches):
            mb_obs = b_obs[mb].to(data.device)
            mb_actions = b_actions[mb].contiguous()
            mb_values = b_values[mb].reshape(-1)
            mb_advantages = b_advantages[mb].reshape(-1)
            mb_returns = b_returns[mb].reshape(-1)

            if hasattr(data.agent, 'lstm'):
                _, newlogprob, entropy, newvalue, lstm_state = data.agent.get_action_and_value(
                    mb_obs, state=lstm_state, action=mb_actions)
                lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
            else:
                _, newlogprob, entropy, newvalue = data.agent.get_action_and_value(
                    mb_obs.reshape(-1, *data.buffers[0].single_observation_space.shape),
                    action=mb_actions,
                )

            logratio = newlogprob - b_logprobs[mb].reshape(-1)
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                old_kls.append(old_approx_kl.item())
                approx_kl = ((ratio - 1) - logratio).mean()
                kls.append(approx_kl.item())
                clipfracs += [
                    ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                ]

            mb_advantages = mb_advantages.reshape(-1)
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_losses.append(pg_loss.item())

            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -config.vf_clip_coef,
                    config.vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
            v_losses.append(v_loss.item())

            entropy_loss = entropy.mean()
            entropy_losses.append(entropy_loss.item())

            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
            data.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(data.agent.parameters(), config.max_grad_norm)
            data.optimizer.step()

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    train_profiler.stop()
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    charts = data.charts
    charts.learning_rate = data.optimizer.param_groups[0]["lr"]

    losses = data.losses
    losses.policy_loss = np.mean(pg_losses)
    losses.value_loss = np.mean(v_losses)
    losses.entropy = np.mean(entropy_losses)
    losses.old_approx_kl = np.mean(old_kls)
    losses.approx_kl = np.mean(kls)
    losses.clipfrac = np.mean(clipfracs)
    losses.explained_variance = explained_var

    perf = data.performance
    perf.total_uptime = int(time.time() - data.start_time)
    perf.total_updates = data.update + 1
    perf.train_time = time.time() - train_time
    perf.train_sps = int(config.batch_size / perf.train_time)
    perf.train_memory = train_profiler.end_mem
    perf.train_pytorch_memory = train_profiler.end_torch_mem
    perf.epoch_time = perf.eval_time + perf.train_time
    perf.epoch_sps = int(config.batch_size / perf.epoch_time)

    if config.verbose:
        print_dashboard(data.stats, data.init_performance, data.performance)

    if data.update % config.checkpoint_interval == 0 or done_training(data):
       save_checkpoint(data)

    data.update += 1

def close(data):
    for envs in data.buffers:
        envs.close()

    if data.wandb is not None:
        artifact_name = f"{data.exp_name}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()

def rollout(env_creator, env_kwargs, model_path, device='cuda', verbose=True):
    env = env_creator(**env_kwargs)
    agent = torch.load(model_path, map_location=device)
    terminal = truncated = True

    import cv2
    bg = cv2.imread('full_map_v2a.png')
 
    while True:
        if terminal or truncated:
            if verbose:
                print('---  Reset  ---')

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        counts_map = env.env.counts_map
        if np.sum(counts_map) > 0 and step % 100 == 0:
            overlay = make_pokemon_red_overlay(bg, counts_map)
            cv2.imshow('Pokemon Red', overlay[1000:][::4, ::4])
            cv2.waitKey(1)

        if verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        if not env_kwargs['headless']:
            env.render()

        step += 1

def done_training(data):
    return data.update >= data.total_updates

def save_checkpoint(data):
    path = os.path.join(data.config.data_dir, data.exp_name)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.update:06d}.pt'
    model_path = os.path.join(path, model_name)

    # Already saved
    if os.path.exists(model_path):
        return model_path

    torch.save(data.agent, model_path)

    state = {
        "optimizer_state_dict": data.optimizer.state_dict(),
        "global_step": data.global_step,
        "agent_step": data.global_step,
        "update": data.update,
        "model_name": model_name,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)

    return model_path

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

def print_dashboard(stats, init_performance, performance):
    output = []
    data = {**stats, **init_performance, **performance}
    
    grouped_data = defaultdict(dict)
    
    for k, v in data.items():
        if k == 'total_uptime':
            v = timedelta(seconds=v)
        if 'memory' in k:
            v = pufferlib.utils.format_bytes(v)
        elif 'time' in k:
            try:
                v = f"{v:.2f} s"
            except:
                pass
        
        first_word, *rest_words = k.split('_')
        rest_words = ' '.join(rest_words).title()
        
        grouped_data[first_word][rest_words] = v
    
    for main_key, sub_dict in grouped_data.items():
        output.append(f"{main_key.title()}")
        for sub_key, sub_value in sub_dict.items():
            output.append(f"    {sub_key}: {sub_value}")
    
    print("\033c", end="")
    print('\n'.join(output))

def make_pokemon_red_overlay(bg, counts):
    nonzero = np.where(counts > 0, 1, 0)
    scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.zeros((*counts.shape, 3))
    hsv[..., 0] = (240.0/360) - scaled*(240.0/360.0)
    hsv[..., 1] = nonzero
    hsv[..., 2] = nonzero

    # Convert the HSV image to RGB
    import matplotlib.colors as mcolors
    overlay = 255*mcolors.hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ones((16, 16, 1), dtype=np.uint8)
    overlay = np.kron(overlay, kernel).astype(np.uint8)
    mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
    mask = np.stack([mask, mask, mask], axis=-1)

    # Offset to align with map
    x_pad, y_pad = 16*16, 16*13
    overlay = np.pad(overlay, ((0, y_pad+8), (0, x_pad), (0, 0)))
    overlay = overlay[y_pad+8:, x_pad:]
    mask = np.pad(mask, ((0, y_pad+8), (0, x_pad), (0, 0)))
    mask = mask[y_pad+8:, x_pad:].astype(bool)

    # Combine with background
    render = bg.copy().astype(np.int32)
    render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render

class CleanPuffeRL:
    __init__ = init
    evaluate = evaluate
    train = train
    close = close
    done_training = done_training
