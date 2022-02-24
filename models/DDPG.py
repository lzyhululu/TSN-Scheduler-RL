""" advantage actor critic """
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .buffer import EpisodesBuffer
from tensorflow.keras.models import load_model
# from .noise import OUActionNoise


class DDPolicyGradient(tf.keras.Model):
    def __init__(self, env, handle, name,
                 batch_size=64, critic_lr=0.01, actor_lr=0.01, reward_decay=0.99, ignore_offsets=False,
                 train_freq=1, target_update=2000, memory_size=2 ** 20, eval_obs=None,
                 use_dueling=True, use_double=True, use_conv=True, sample_buffer_capacity=500,
                 num_gpu=1, nums_all_agent=None, network_type=0):
        """init a model"""
        super().__init__(self)
        # ======================== set config  ========================
        self.batch_size = batch_size
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_agents = env.get_num(handle)
        self.ignore_offsets = ignore_offsets
        self.num_actions = env.get_action_space(handle)[0] if not self.ignore_offsets else 1
        self.reward_decay = reward_decay
        self.nums_all_agent = nums_all_agent
        self.handle = handle.value
        # ======================= build network =======================
        # initialize input dimensions
        self.input_view = self.view_space[0]
        self.input_feature = self.feature_space[0]

        # Creating Optimizer for actor and critic networks
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

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
        # inputs = layers.Input(shape=(self.input_view + self.input_feature*self.num_agents,))
        inputs = layers.Input(shape=(self.input_feature * self.num_agents,))
        out = layers.Dense(128, activation="selu", kernel_initializer="lecun_normal")(inputs)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(128, activation="selu", kernel_initializer="lecun_normal")(out)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)

        # using sigmoid activation as action values for
        # for our environment lies between 0 to 1
        outputs = layers.Dense(self.num_actions*self.num_agents,
                               activation="sigmoid", kernel_regularizer=last_init)(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def _get_critic(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        num = sum(self.nums_all_agent)

        # State as input, here this state is the observation of all the agents (input_view)
        # hence this state will have information of observation of all the agents
        # state_input = layers.Input(shape=(self.input_view + self.input_feature*num,))
        state_input = layers.Input(shape=(self.input_feature * num,))
        state_out = layers.Dense(16, activation="selu", kernel_initializer="lecun_normal")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(state_out)
        state_out = layers.BatchNormalization()(state_out)

        # Action all the agents as input
        action_input1 = layers.Input(shape=(self.nums_all_agent[0]*self.num_actions,))
        action_input2 = layers.Input(shape=(self.nums_all_agent[1]*self.num_actions,))
        action_input3 = layers.Input(shape=(self.nums_all_agent[2]*self.num_actions,))
        action_input4 = layers.Input(shape=(self.nums_all_agent[3]*self.num_actions,))

        action_input = layers.Concatenate()([action_input1, action_input2, action_input3, action_input4])
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
        model = tf.keras.Model([state_input, action_input1, action_input2, action_input3, action_input4], outputs)
        return model

    def sample_step(self, ids, obs, acts, next_obs, rewards):
        """record a step"""
        self.sample_buffer.record_step(ids, obs, acts, next_obs, rewards, self.num_actions, self.ignore_offsets)

    def infer_action(self, raw_obs, *args, **kwargs):
        """infer action for a batch of agents
        bahaviour policy: beta, which combines online policy muon and ou_noises
        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        view, feature = raw_obs[0], raw_obs[1]
        feature = tf.reshape(feature, [1, -1])
        # Get actions for each agents from respective models and store them in list
        # input_para = np.concatenate((view, feature), axis=1)
        input_para = feature
        actions = self.ac_model(input_para)
        return actions

    def train_first(self, sample_buffer, batch_indices):
        """
        first means to get the next_actions
        feed new data sample and train
        sample_buffer: buffer.EpisodesBuffer-p
            buffer contains samples

        target_action:
            next_actions predict
        """
        # train

        # Convert to tensors
        # feature_batch
        feature_batch = tf.convert_to_tensor(np.array(sample_buffer.total_features)[batch_indices])
        feature_batch = tf.reshape(feature_batch, [self.batch_size, -1])
        # state_batch
        # view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_view)[batch_indices])
        # view_batch = tf.squeeze(view_batch, axis=1)
        # state_batch = tf.concat([view_batch, feature_batch], axis=1)
        # action_batch
        action_batch = tf.convert_to_tensor(np.array(sample_buffer.total_actions)[batch_indices])
        action_batch = tf.squeeze(action_batch, axis=1)
        # reward_batch
        reward_batch = tf.convert_to_tensor(np.array(sample_buffer.total_rewards)[batch_indices])
        reward_batch = tf.expand_dims(reward_batch, axis=1)
        # next_state_batch
        # next_view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_next_view)[batch_indices])
        # next_view_batch = tf.squeeze(next_view_batch, axis=1)
        next_feature_batch = tf.convert_to_tensor(np.array(sample_buffer.total_next_features)[batch_indices])
        next_feature_batch = tf.reshape(next_feature_batch, [self.batch_size, -1])
        # next_state_batch = tf.concat([next_view_batch, next_feature_batch], axis=1)

        # Training  and Updating ***critic model***
        # target_action = self.target_ac(next_state_batch)
        target_action = self.target_ac(next_feature_batch)
        # Updating and Training of ***actor network**
        # actions = self.ac_model(state_batch)
        actions = self.ac_model(feature_batch)

        return target_action, feature_batch, action_batch, reward_batch, next_feature_batch, actions

    def train_second(self, sample_buffer, batch_indices, target_actions, features_batch, actions_batch, rewards_batch,
                     next_features_batch, last_actions_batch):
        """
        train online Q networks
        """
        # train
        # total information
        # state total
        # view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_view)[batch_indices])
        # view_batch = tf.squeeze(view_batch, axis=1)
        features = tf.concat([features_batch[i] for i in range(len(self.nums_all_agent))], axis=1)
        # state_batch = tf.concat([view_batch, features], axis=1)

        # state' total
        # next_view_batch = tf.convert_to_tensor(np.array(sample_buffer.total_next_view)[batch_indices])
        # next_view_batch = tf.squeeze(next_view_batch, axis=1)
        next_features = tf.concat([next_features_batch[i] for i in range(len(self.nums_all_agent))], axis=1)
        # next_state_batch = tf.concat([next_view_batch, next_features], axis=1)

        # rewards last action
        rewards = tf.concat([rewards_batch[i] for i in range(len(self.nums_all_agent))], axis=1)
        rewards = tf.reduce_sum(rewards, axis=1, keepdims=True)
        # Finding Gradient of loss function
        with tf.GradientTape() as tape:
            # y = self.reward_decay * self.target_cr([next_state_batch, next_actions])
            # y = rewards + self.reward_decay * self.target_cr([next_state_batch, target_actions[0], target_actions[1],
            #                                                   target_actions[2], target_actions[3]])
            y = rewards + self.reward_decay * self.target_cr([next_features, target_actions[0], target_actions[1],
                                                              target_actions[2], target_actions[3]])
            # state_batch
            critic_value = self.cr_model([features, actions_batch[0], actions_batch[1], actions_batch[2],
                                          actions_batch[3]])

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.cr_model.trainable_variables)

        # Applying gradients to update critic network of ith agent
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.cr_model.trainable_variables))
        # Updating and training of ***online critic network*** ended

        # Finding gradient of actor model
        # feature_batch
        feature_batch_self = tf.convert_to_tensor(np.array(sample_buffer.total_features)[batch_indices])
        feature_batch_self = tf.reshape(feature_batch_self, [self.batch_size, -1])
        # state_batch this agent
        # state_batch_self = tf.concat([view_batch, feature_batch_self], axis=1)
        last_actions_batch = [tf.expand_dims(k, axis=1) for k in last_actions_batch]
        with tf.GradientTape(persistent=True) as tape:
            # [state_batch_self[0]]
            action_ = self.ac_model(np.array([feature_batch_self[0]]))
            # actions last time with muon policy
            last_actions = [last_actions_batch[i][0] if i != self.handle else action_ for i in range(
                len(self.nums_all_agent))]
            # cr_input = [tf.expand_dims(state_batch[0], axis=0)] + last_actions
            cr_input = [tf.expand_dims(features[0], axis=0)] + last_actions
            critic_value = self.cr_model(cr_input)

        critic_grad = tape.gradient(critic_value, action_)
        actor_grad = tape.gradient(action_, self.ac_model.trainable_variables)

        new_actor_grad = [critic_grad[0][0] * element for element in actor_grad]

        for k in range(1, self.batch_size):
            with tf.GradientTape(persistent=True) as tape:
                # [state_batch_self[k]]
                action_ = self.ac_model(np.array([feature_batch_self[k]]))
                # actions last time with muon policy
                last_actions = [last_actions_batch[i][k] if i != self.handle else action_ for i in
                                range(len(self.nums_all_agent))]
                # cr_input = [tf.expand_dims(state_batch[k], axis=0)] + last_actions
                cr_input = [tf.expand_dims(features[k], axis=0)] + last_actions
                critic_value = self.cr_model(cr_input)

            critic_grad = tape.gradient(critic_value, action_)
            actor_grad = tape.gradient(action_, self.ac_model.trainable_variables)

            for j in range(len(new_actor_grad)):
                new_actor_grad[j] = new_actor_grad[j] + critic_grad[0][0] * actor_grad[j]

        # Updating gradient network if it is 1st agent
        new_actor_grad = [-1 * element / self.batch_size for element in new_actor_grad]
        self.actor_optimizer.apply_gradients(zip(new_actor_grad, self.ac_model.trainable_variables))

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    def update_target(self, tau):
        new_weights = []
        target_variables = self.target_cr.weights

        for j, variable in enumerate(self.cr_model.weights):
            new_weights.append(variable * tau + target_variables[j] * (1 - tau))

        self.target_cr.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_ac.weights

        for j, variable in enumerate(self.ac_model.weights):
            new_weights.append(variable * tau + target_variables[j] * (1 - tau))

        self.target_ac.set_weights(new_weights)

    def save(self, dir_name, epoch):
        """save model to dir

        Parameters
        ----------
        dir_name: str
            name of the directory
        epoch: int
        """
        self.ac_model.save_weights(dir_name + '/actor' + str(self.handle) + '_' + str(epoch) + '.h5')
        self.cr_model.save_weights(dir_name + '/critic' + str(self.handle) + '_' + str(epoch) + '.h5')
        self.target_ac.save_weights(dir_name + '/target_actor' + str(self.handle) + '_' + str(epoch) + '.h5')
        self.target_cr.save_weights(dir_name + '/target_critic' + str(self.handle) + '_' + str(epoch) + '.h5')

    def load(self, dir_name, epoch=0):
        """save model to dir

        Parameters
        ----------
        dir_name: str
            name of the directory
        epoch: int
        """
        self.ac_model.load_weights(dir_name + '/actor' + str(self.handle) + '_' + str(epoch) + '.h5')
        self.cr_model.load_weights(dir_name + '/critic' + str(self.handle) + '_' + str(epoch) + '.h5')
        self.target_ac.load_weights(dir_name + '/target_actor' +
                                    str(self.handle) + '_' + str(epoch) + '.h5')
        self.target_cr.load_weights(dir_name + '/target_critic' +
                                    str(self.handle) + '_' + str(epoch) + '.h5')
