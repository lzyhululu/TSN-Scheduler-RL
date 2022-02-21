""" advantage actor critic """
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .buffer import ReplayBuffer, EpisodesBuffer


class DDPolicyGradient(tf.keras.Model):
    def __init__(self, env, handle, name,
                 batch_size=64, critic_lr=1e-4, actor_lr=5e-5, reward_decay=0.99,
                 train_freq=1, target_update=2000, memory_size=2 ** 20, eval_obs=None,
                 use_dueling=True, use_double=True, use_conv=True, sample_buffer_capacity=1000,
                 num_gpu=1, infer_batch_size=8192, network_type=0):
        """init a model"""
        super().__init__(self)
        # ======================== set config  ========================
        self.batch_size = batch_size
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_agents = env.get_num(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.reward_decay = reward_decay
        # ======================= build network =======================
        # initialize input dimensions
        self.input_view = self.view_space[0]
        self.input_feature = self.feature_space[0]

        # Initializing basic actor/critic models
        # Neural Net Models for agents will be saved in there
        self.ac_model = self._get_actor()
        self.cr_model = self._get_critic()
        self.target_ac = self._get_actor()
        self.target_cr = self._get_critic()

        # Making the weights equal initially
        self.target_ac.set_weights(self.ac_model.get_weights())
        self.target_cr.set_weights(self.cr_model.get_weights())

        # init episiodes buffer
        self.sample_buffer = EpisodesBuffer(capacity=sample_buffer_capacity)

    def _get_actor(self):
        """config the actor network"""

        # Initialize weights between -3e-5 and 3e-5
        last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

        # Actor will get observation of the agent, shared in the map
        inputs = layers.Input(shape=(self.input_view + self.input_feature*self.num_agents,))
        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(inputs)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)

        # using sigmoid activation as action values for
        # for our environment lies between 0 to 1
        outputs = layers.Dense(self.num_actions*self.num_agents, activation="sigmoid", kernel_regularizer=last_init)(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def _get_critic(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # State as input, here this state is the observation of all the agents (input_view)
        # hence this state will have information of observation of all the agents
        state_input = layers.Input(shape=(self.input_view,))
        state_out = layers.Dense(16, activation="selu", kernel_initializer="lecun_normal")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(state_out)
        state_out = layers.BatchNormalization()(state_out)

        # All the agents actions as input
        action_input = layers.Input(shape=(self.input_feature*self.num_agents,))
        action_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(action_input)
        action_out = layers.BatchNormalization()(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(concat)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(out)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)

        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def sample_step(self, ids, obs, acts, next_obs, rewards):
        """record a step"""
        self.sample_buffer.record_step(ids, obs, acts, next_obs, rewards, self.num_actions)

    def infer_action(self, raw_obs, ids, *args, **kwargs):
        """infer action for a batch of agents

        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)
        ids: numpy array
            ids of agents

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        view, feature = raw_obs[0], raw_obs[1]
        feature = tf.reshape(feature, [1, -1])
        # Get actions for each agents from respective models and store them in list
        input_para = np.concatenate((view, feature), axis=1)
        actions = self.ac_model(input_para)
        return actions

    def train_first(self, sample_buffer, print_every=1000):
        """
        first means to get the next_actions
        feed new data sample and train
        sample_buffer: buffer.EpisodesBuffer-p
            buffer contains samples

        Returns
        -------
        loss: list
            policy gradient loss, critic loss, entropy loss
        value: float
            estimated state value
        """
        # train
        # Updating networks of all the agents
        # by looping over number of agents
        for i in range(self.num_agents):

            # Get sampling range
            record_range = min(sample_buffer.capacity, sample_buffer.counter())

            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            # state_batch
            view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_view)[batch_indices])
            feature_batch = tf.convert_to_tensor(np.array(sample_buffer.total_features)[batch_indices])
            feature_batch = tf.reshape(feature_batch, [self.batch_size, 1, -1])
            state_batch = tf.concat([view_batch, feature_batch], axis=2)
            state_batch = tf.squeeze(state_batch, axis=1)
            # action_batch
            action_batch = tf.convert_to_tensor(np.array(sample_buffer.total_actions)[batch_indices])
            # reward_batch
            reward_batch = tf.convert_to_tensor(np.array(sample_buffer.total_rewards)[batch_indices])
            # next_state_batch
            next_view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_next_view)[batch_indices])
            next_feature_batch = tf.convert_to_tensor(np.array(sample_buffer.total_next_features)[batch_indices])
            next_feature_batch = tf.reshape(next_feature_batch, [self.batch_size, 1, -1])
            next_state_batch = tf.concat([next_view_batch, next_feature_batch], axis=2)
            next_state_batch = tf.squeeze(next_state_batch, axis=1)

            # Training  and Updating ***critic model*** of ith agent
            target_action = self.target_ac(next_state_batch)
        return target_action
            # Finding Gradient of loss function
            with tf.GradientTape() as tape:
                y = reward_batch[:, i] + self.reward_decay * target_cr[i]([
                    next_state_batch, target_action_batch1,
                    target_action_batch2, target_action_batch3
                ])

                critic_value = cr_models[i]([
                    state_batch, action_batch1, action_batch2, action_batch3
                ])

                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, cr_models[i].trainable_variables)

            # Applying gradients to update critic network of ith agent
            critic_optimizer.apply_gradients(
                zip(critic_grad, cr_models[i].trainable_variables)
            )
            # Updating and training of ***critic network*** ended

            # Updating and Training of ***actor network** for ith agent
            actions = np.zeros((self.batch_size, self.num_agents))
            for j in range(self.num_agents):
                a = ac_models[j](state_batch[:, 5 * j:5 * (j + 1)])
                actions[:, j] = tf.reshape(a, [self.batch_size])

            # Finding gradient of actor model if it is 1st agent
            if i == 0:

                with tf.GradientTape(persistent=True) as tape:

                    action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][0]]))

                    critic_value = cr_models[i]([np.array([state_batch[0]]), action_, np.array([actions[:, 1][0]]),
                                                 np.array([actions[:, 2][0]])])

                critic_grad = tape.gradient(critic_value, action_)
                actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                new_actor_grad = [critic_grad[0][0] * element for element in actor_grad]

                for k in range(1, self.batch_size):
                    with tf.GradientTape(persistent=True) as tape:

                        action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][k]]))

                        critic_value = cr_models[i]([np.array([state_batch[k]]), action_, np.array([actions[:, 1][k]]),
                                                     np.array([actions[:, 2][k]])])

                    critic_grad = tape.gradient(critic_value, action_)
                    actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                    for l in range(len(new_actor_grad)):
                        new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0] * actor_grad[l]

                # Updating gradient network if it is 1st agent
                new_actor_grad = [-1 * element / self.batch_size for element in new_actor_grad]
                actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))

            # Finding gradient of actor model if it is 2nd agent
            elif i == 1:
                with tf.GradientTape(persistent=True) as tape:

                    action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][0]]))

                    critic_value = cr_models[i]([np.array([state_batch[0]]), np.array([actions[:, 0][0]]), action_,
                                                 np.array([actions[:, 2][0]])])

                critic_grad = tape.gradient(critic_value, action_)
                actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                new_actor_grad = [critic_grad[0][0] * element for element in actor_grad]

                for k in range(1, self.batch_size):
                    with tf.GradientTape(persistent=True) as tape:

                        action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][k]]))

                        critic_value = cr_models[i]([np.array([state_batch[k]]), np.array([actions[:, 0][k]]), action_,
                                                     np.array([actions[:, 2][k]])])

                    critic_grad = tape.gradient(critic_value, action_)
                    actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                    for l in range(len(new_actor_grad)):
                        new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0] * actor_grad[l]

                # Updating gradient network if it is 2nd agent
                new_actor_grad = [-1 * element / self.batch_size for element in new_actor_grad]
                actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))

            # Finding gradient of actor model if it is 3rd agent
            else:
                with tf.GradientTape(persistent=True) as tape:

                    action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][0]]))

                    critic_value = cr_models[i]([np.array([state_batch[0]]), np.array([actions[:, 0][0]]),
                                                 np.array([actions[:, 1][0]]), action_])

                critic_grad = tape.gradient(critic_value, action_)
                actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                new_actor_grad = [critic_grad[0][0] * element for element in actor_grad]

                for k in range(1, self.batch_size):
                    with tf.GradientTape(persistent=True) as tape:

                        action_ = ac_models[i](np.array([state_batch[:, 5 * i:5 * (i + 1)][k]]))

                        critic_value = cr_models[i]([np.array([state_batch[k]]), np.array([actions[:, 0][k]]),
                                                     np.array([actions[:, 1][k]]), action_])

                    critic_grad = tape.gradient(critic_value, action_)
                    actor_grad = tape.gradient(action_, ac_models[i].trainable_variables)

                    for l in range(len(new_actor_grad)):
                        new_actor_grad[l] = new_actor_grad[l] + critic_grad[0][0] * actor_grad[l]

                # Updating gradient network if it is 3rd agent
                new_actor_grad = [-1 * element / self.batch_size for element in new_actor_grad]
                actor_optimizer.apply_gradients(zip(new_actor_grad, ac_models[i].trainable_variables))

        return [pg_loss, vf_loss, ent_loss], np.mean(state_value)

    def get_info(self):
        """
        get information of the model
        """
        return "a2c train_time: %d" % (self.train_ct)
