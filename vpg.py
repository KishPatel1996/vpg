#Cribbed from Open AI's spinning up: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
from typing import List, Callable
import tensorflow as tf
import numpy as np
import gym
from time import sleep



def fcn(x: tf.Tensor, hidden_size: List[int], action_size: int, activation_layer=tf.nn.relu, name_scope="Actor"):
    for dim in hidden_size:
        x = tf.layers.dense(x, units=dim, activation=activation_layer)
    return tf.layers.dense(x, units=action_size, activation=None)


class VanillaPolicyGradient():
    def __init__(self,
                 env,
                 actor_net_func: Callable[[tf.Tensor, List[int], int], tf.Tensor],
                 num_epochs=300,
                 sample_size=5000,
                 learning_rate=1e-3):
        self.env = env
        self.actor_net_func = actor_net_func
        self.num_epochs = num_epochs
        self.sample_size = sample_size
        self.learning_rate = learning_rate

        #initalize actor net with input placeholders
        self.actor_obs_ph = tf.placeholder(shape=(None,
                                                  self.env.observation_space.shape[0]),
                                           dtype=tf.float32,
                                           name="observations_ph")
        self.actor_net = self.actor_net_func(self.actor_obs_ph,
                                             [16, 16, 16],
                                             self.env.action_space.n)
        self.actor_action = tf.squeeze(tf.multinomial(logits=self.actor_net, num_samples=1), axis=1)
        self.trained = False
        self.sess = None

    def run_single_iteration(self, with_render: bool=False):
        assert self.trained, "Net has not been trained, run .train() first."
        episode_observations = []
        episode_rewards = []
        obs = self.env.reset()
        is_done = False
        while not is_done:
            if with_render:
                env.render()
                sleep(0.03)
            episode_observations.append(obs.copy())

            action =  self.sess.run(self.actor_action,
                                               {self.actor_obs_ph: obs.reshape(1,-1)})[0]
            obs, reward, is_done, _ = env.step(action)
            episode_rewards.append(reward)
        return episode_observations, episode_rewards




    @staticmethod
    def _future_rewards(rewards: List[int]) -> np.ndarray:
        future_rewards = np.zeros(len(rewards))
        for i in reversed(range(len(rewards))):
            future_rewards[i] = (future_rewards[i+1] + rewards[i]
                                 if i < len(rewards) - 1
                                 else rewards[i])
        return future_rewards

    def train(self) -> None:


        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name="actions_ph")
        weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="weights_ph")
        act_mask = tf.one_hot(act_ph, self.env.action_space.n)
        logs = tf.reduce_sum(act_mask * tf.nn.log_softmax(self.actor_net), axis=1)
        loss = -1 * tf.reduce_mean(logs * weights_ph)

        train_op = (tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    .minimize(loss)
                    )
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


        batch_actions = []
        batch_rewards = []


        for i in range(self.num_epochs):

            batch_actions = []
            batch_rewards = []
            batch_obs = []
            average_reward = 0
            total_runs = 0
            average_len = 0
            while len(batch_actions) < self.sample_size:
                obs = self.env.reset()
                episode_rewards = []
                episode_actions = []
                is_done = False
                while not is_done:
                    batch_obs.append(obs.copy())
                    net_action = self.sess.run(self.actor_action, {self.actor_obs_ph: obs.reshape(1,-1)})[0]

                    obs, reward, is_done, _ = self.env.step(net_action)

                    episode_actions.append(net_action)
                    episode_rewards.append(reward)

                #Episode be done by the time it arrives here
                episode_rewards = VanillaPolicyGradient._future_rewards(episode_rewards)
                batch_actions.extend(episode_actions)
                batch_rewards.extend(list(episode_rewards))

                total_runs += 1
                average_reward = average_reward + (1/ total_runs) * (episode_rewards[-1] - average_reward)
                average_len = average_len + (1/ total_runs) * (len(episode_actions) - average_len)

            #enough data points to backprop
            self.sess.run(train_op, feed_dict={
                act_ph: np.array(batch_actions),
                weights_ph: np.array(batch_rewards),
                self.actor_obs_ph: np.array(batch_obs)
                })
            print("Epoch: {} --  Average Ep Reward: {} -- Average Ep Length: {}".format(i, average_reward, average_len))
        self.trained = True


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    vpg = VanillaPolicyGradient(env, actor_net_func=fcn)
    vpg.train()
    vpg.run_single_iteration(with_render=True)
