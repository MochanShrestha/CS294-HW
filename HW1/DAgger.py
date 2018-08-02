
# Get some data from the expert to get started

import tensorflow as tf
import gym.spaces
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

import hw1files
import hw1files.tf_util
import hw1files.load_policy
from hw1files.run_expert import expert_actions
from hw1files import load_policy


# Function to shuffle the weights

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def main():
    #expert_policy_file = 'hw1files/experts/Hopper-v1.pkl'
    #envname = 'Hopper-v2'
    #expert_policy_file = 'hw1files/experts/Ant-v1.pkl'
    #envname = 'Ant-v2'
    #expert_policy_file = 'hw1files/experts/Reacher-v1.pkl'
    #envname = 'Reacher-v2'
    expert_policy_file = 'hw1files/experts/Walker2d-v1.pkl'
    envname = 'Walker2d-v2'
    max_timesteps = 500
    num_rollouts = 10
    render = False

    env = gym.make(envname)

    # Get the set of observations by running our supervised

    def get_observations(num_rollouts, render=False):
        # Run the supervised model and produce the reward function
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        timesteps = []
        actions = []
        for i in range(num_rollouts):
            # tf_util.initialize()
            if i % 10 == 0: print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                # action = policy_fn(obs[None,:])
                action = model.predict(obs.reshape(1, obs_len))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break

            returns.append(totalr)
            timesteps.append(steps)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        print('mean timesteps', np.mean(timesteps))
        print('std of timesteps', np.std(timesteps))

        return observations

    data = hw1files.run_expert.expert_actions(expert_policy_file, envname, max_timesteps, num_rollouts, render)

    expert_observations = data['observations']
    expert_actions = data['actions']
    act_shape = expert_actions.shape
    expert_actions = expert_actions.reshape([expert_actions.shape[0], expert_actions.shape[2]])

    obs_len = len(expert_observations[0])
    action_len = len(expert_actions[0])

    print(expert_observations.shape)
    print(expert_actions.shape)


    # Setup the tensorflow model

    # Create a neural network that takes in the right number of inputs and produces the right number of outputs

    model = Sequential()
    model.add(Dense(121, input_dim=obs_len, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(29, activation='relu'))
    keras.layers.Dropout(0.2, noise_shape=None, seed=None)
    model.add(Dense(11, activation='relu'))
    model.add(Dense(action_len))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    initial_weights = model.get_weights()



    # Policy function

    policy_fn = load_policy.load_policy(expert_policy_file)

    # Do the DAgger

    all_obs = expert_observations
    all_actions = expert_actions

    for i in range(1, 100):
        new_obs = get_observations(5, True)

        new_actions = []
        with tf.Session():
            # tf_util.initialize()
            for ob in new_obs:
                ac = policy_fn(ob[None, :])
                new_actions.append(ac)

        all_obs = np.append(all_obs, np.array(new_obs), 0)
        all_actions = np.append(all_actions, np.squeeze(np.array(new_actions)), 0)

        # Train using the new model
        print(all_obs.shape)
        print(all_actions.shape)
        #shuffle_weights(model, initial_weights)
        model.fit(all_obs, all_actions, epochs=20, batch_size=100)

if __name__ == '__main__':
    main()
