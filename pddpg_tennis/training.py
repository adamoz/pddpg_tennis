from collections import deque
import numpy as np
import os
import pickle


def pddpg_training(agent, env, save_path, n_episodes=2000, n_max_steps_per_episode=10000, logger=None):
    """Deep Q-Learning.

    Params:
        n_episodes (int): maximum number of training episodes
        n_max_steps_per_episode (int): maximum number of timesteps per episode

    """
    printer = print
    if logger is not None:
        printer = logger.info

    meta = {
        'scores': [],
        'meta_path': os.path.join(save_path, 'meta.pickle'),
        'model_path': os.path.join(save_path, 'model'),
        'agent': repr(agent)
    }

    scores_window = deque(maxlen=100)  # last 100 scores
    best_mean_score = 0

    for i_episode in range(1, n_episodes + 1):
        states = env.reset()
        agent.reset()
        score = np.zeros([env.num_agents])

        for step in range(n_max_steps_per_episode):
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        raw_score = score
        score = np.max(score, axis=0)
        scores_window.append(score)
        meta['scores'].append(score)
        mean_score = np.mean(scores_window)

        printer(f'\rEpisode {i_episode}\tAverage Score: {mean_score}\tScore {raw_score}')
        if mean_score >= best_mean_score:
            best_mean_score = mean_score
            agent.save(meta['model_path'])
            with open(meta['meta_path'], 'wb') as wb:
                pickle.dump(meta, wb)
        if best_mean_score >= 0.5:
            break

    with open(meta['meta_path'], 'wb') as wb:
        pickle.dump(meta, wb)
    printer(f'\rWe have achieved {best_mean_score} mean score (0.5+ is good result)')
    return meta
