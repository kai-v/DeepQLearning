#!/usr/bin/env python 3.8
# -*- coding: utf-8 -*-
"""

@author: kai
"""

from email import policy
from tf_agents.drivers import py_driver
import tensorflow as tf
import numpy as np
import IPython
import datetime

# import reverb
from tf_agents.trajectories import trajectory
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, BoundedArraySpec, tensor_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.replay_buffers import reverb_replay_buffer, TFUniformReplayBuffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_eager_policy
import os
from tf_agents.networks import q_network, categorical_q_network
from tensorflow.python.client import device_lib
import tensorflow.keras.models
from tf_agents.utils.common import Checkpointer
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
import rospy
import matplotlib
import matplotlib.pyplot as plt

#

"""Used tensorflow checkpoint to save checkpoints in the model training so it can be restored"""


class DeepQLearning:
    def __init__(self, env):

        self.env = env
        self.train_env = env
        self.eval_env = env
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #self.model_location = "ckpts"
        self.test_mode = True

    # Metrics and evaluation

    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def Learn(self):

        num_iterations = 510  # @param {type:"integer"}
        epsilon_decay = 500
        store_checkpoint = 500

        initial_collect_steps = 200  # @param {type:"integer"}
        collect_steps_per_iteration = 10  # @param {type:"integer"}
        replay_buffer_max_length = 100000  # @param {type:"integer"}

        batch_size = 64  # @param {type:"integer"}
        learning_rate = 0.00025  # @param {type:"number"}
        log_interval = 500  # @param {type:"integer"}

        num_eval_episodes = 10  # @param {type:"integer"}
        eval_interval = 100  # @param {type:"integer"}

        train_env = tf_py_environment.TFPyEnvironment(self.train_env)
        print("initialised train env")
        eval_env = tf_py_environment.TFPyEnvironment(self.eval_env)
        print("initialised test env")

        print("time step spec:")
        print(train_env.time_step_spec())

        print("Action Spec:")
        print(train_env.action_spec())

        time_step = self.env.reset()
        print("Time step:")

        # train_env.close()

        fc_layer_params = (256, 256)
        action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
        # num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        """ Initialise distributed Q net  """

        # if self.test_mode == False:
        q_net = q_network.QNetwork(
            train_env.observation_spec(),
            action_spec=train_env.action_spec(),
            fc_layer_params=fc_layer_params,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_step_counter = tf.Variable(0)

        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.9,
            decay_steps=epsilon_decay,
            end_learning_rate=0.01,
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
        )

        agent.initialize()
        collect_policy = agent.collect_policy

        random_policy = random_tf_policy.RandomTFPolicy(
            train_env.time_step_spec(), train_env.action_spec()
        )

        replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_max_length,
        )

        policy_dir = "policy"
        checkpoint_dir = "ckpts"

        train_checkpointer = Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=5,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )


        def collect_step(environment, policy):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)

        def test_steps(environment, policy):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)

        if self.test_mode == False:

            for _ in range(initial_collect_steps):
                collect_step(train_env, random_policy)

            dataset = replay_buffer.as_dataset(
                num_parallel_calls=2, sample_batch_size=batch_size, num_steps=2
            ).prefetch(3)

                # dataset
            iterator = iter(dataset)
                # print(iterator)

            agent.train = common.function(agent.train)

                # Reset the train step.
            agent.train_step_counter.assign(0)

                # Evaluate the agent's policy once before training.
            avg_return = self.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            returns = [avg_return]

        """ If in test mode restore from last check point """
        if self.test_mode == True:

            train_checkpointer.initialize_or_restore()
            global_step = tf.compat.v1.train.get_global_step()
            saved_policy = tf.saved_model.load(policy_dir)


            print("Restored from checkpoint")

            """Testing Loop"""

            for _ in range(1000):

                for _ in range(collect_steps_per_iteration):
                    test_steps(eval_env, saved_policy)

                # experience, unused_info = next(iterator)
                #train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()

                if step % 10 == 0:
                    print("step = {0}: ".format(step))


        else:

            """Learning Loop"""

            for _ in range(num_iterations):

                for _ in range(collect_steps_per_iteration):
                    collect_step(train_env, collect_policy)

                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()

                if step % log_interval == 0:
                    print("step = {0}: loss = {1}".format(step, train_loss))

                if step % eval_interval == 0:
                    avg_return = self.compute_avg_return(
                        eval_env, agent.policy, num_eval_episodes
                    )
                    print(
                        "step = {0}: Average Return = {1:.2f}".format(step, avg_return)
                    )
                    returns.append(avg_return)

                if step == store_checkpoint:
                    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

                    # train_checkpointer = Checkpointer(
                    #     ckpt_dir=checkpoint_dir,
                    #     max_to_keep=5,
                    #     agent=agent,
                    #     policy=agent.policy,
                    #     replay_buffer=replay_buffer
                    # )

                    train_checkpointer.initialize_or_restore()
                    train_checkpointer.save(step)
                    tf_policy_saver.save(policy_dir)


            # print("Done Learning, saving the model ...")
