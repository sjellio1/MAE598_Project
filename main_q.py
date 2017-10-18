### Test Q learning and AC

from pylab import *
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import rocketenv


class Agent(object):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, input_size=6, hidden_size=2, gamma=0.95,
                 action_size=6, lr=0.1, dir='tmp/trial/'):
        # call the cartpole simulator from OpenAI gym package
        self.env = rocketenv.RocketEnv()
        # If you wish to save the simulation video, simply uncomment the line below
        # self.env = wrappers.Monitor(self.env, dir, force=True, video_callable=self.video_callable)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.action_size = action_size
        self.lr = lr
        # save the hyper parameters
        self.params = self.__dict__.copy()

        # inputs to the controller
        self.state_pl = tf.placeholder(tf.float32, [None, input_size])
        self.action_pl = tf.placeholder(tf.int32, [None])
        self.reward_pl = tf.placeholder(tf.float32, [None])
        self.Q_next = tf.placeholder(tf.float32, [None, input_size])

        # Here we use a single layered neural network as controller, which proved to be sufficient enough.
        # More complicated ones can be plugged in as well.
        # hidden_layer = layers.fully_connected(self.state_pl,
        #                                      hidden_size,
        #                                      biases_initializer=None,
        #                                      activation_fn=tf.nn.relu)
        hidden_layer = layers.fully_connected(self.state_pl,
                                              hidden_size,
                                              biases_initializer=None,
                                              activation_fn=tf.nn.relu)
        self.Q = layers.fully_connected(hidden_layer,
                                             action_size,
                                             biases_initializer=None,
                                             activation_fn=tf.nn.relu)
        self.output = tf.nn.softmax(self.Q)

        # update Q values
        self.y = self.reward_pl + tf.reduce_max(self.Q_next)

        # responsible output
        self.one_hot = tf.one_hot(self.action_pl, action_size)
        self.actual_Q = tf.reduce_sum(self.Q * self.one_hot, axis=1)

        # Q learning
        self.loss = (self.y-self.actual_Q)**2

        # get all network variables
        variables = tf.trainable_variables()
        self.variable_pls = []
        for i, var in enumerate(variables):
            self.variable_pls.append(tf.placeholder(tf.float32))

        # compute the gradient values
        self.gradients = tf.gradients(self.loss, variables)

        # update network variables
        solver = tf.train.AdamOptimizer(learning_rate=self.lr)
        # solver = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.95)
        self.update = solver.apply_gradients(zip(self.variable_pls, variables))


    def video_callable(self, episode_id):
        # display the simulation trajectory every 50 epoch
        return episode_id % 50 == 0

    def next_action(self, sess, feed_dict, greedy=False):
        """Pick an action based on the current state.
        Args:
        - sess: a tensorflow session
        - feed_dict: parameter for sess.run()
        - greedy: boolean, whether to take action greedily
        Return:
            Integer, action to be taken.
        """
        ans = sess.run(self.output, feed_dict=feed_dict)[0]
        if greedy:
            return ans.argmax()
        else:
            return np.random.choice(range(self.action_size), p=ans)

    def show_parameters(self):
        """Helper function to show the hyper parameters."""
        for key, value in self.params.items():
            print(key, '=', value)


def discounted_reward(rewards, gamma):
    """Compute the discounted reward."""
    ans = np.zeros_like(rewards)
    running_sum = 0
    # weight = np.array(range(len(rewards)))
    # for i in weight:
    #     weight[i] = weight[i]**2
    # reward_weighted = rewards * np.array(weight)
    # reward_weighted = np.zeros(len(rewards))
    # reward_weighted[-1] = rewards[-1]
    # compute the result backward
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        ans[i] = running_sum
    # ans += 1000*rewards[-1]
    return ans


def one_trial(agent, sess, grad_buffer, reward_itr, i, render=False):
    '''
    this function does follow things before a trial is done:
    1. get a sequence of actions based on the current state and a given control policy
    2. get the system response of a given action
    3. get the instantaneous reward of this action
    once a trial is done:
    1. get the "long term" value of the controller
    2. get the gradient of the controller
    3. update the controller variables
    4. output the state history
    '''

    # reset the environment
    s = agent.env.reset()
    for idx in range(len(grad_buffer)):
        grad_buffer[idx] *= 0
    state_history = []
    reward_history = []
    action_history = []
    current_reward = 0

    while True:

        feed_dict = {agent.state_pl: [s]}
        # update the controller deterministically
        greedy = False
        # get the controller output under a given state
        action = agent.next_action(sess, feed_dict, greedy=greedy)
        # get the next states after taking an action
        s_next, r, done, _ = agent.env.step([action, len(state_history)])

        current_reward += r
        # current_reward = r

        state_history.append(s)
        reward_history.append(r)
        action_history.append(action)

        # best Q at s_next
        Q_next = sess.run(agent.Q, {agent.state_pl: [s_next]}) * (1. - done)

        # update network gradients at every time step
        feed_dict = {
            agent.reward_pl: [r],
            agent.action_pl: [action],
            agent.state_pl: [s],
            agent.Q_next: Q_next
        }
        g = sess.run(agent.gradients, feed_dict=feed_dict)
        for idx, grad in enumerate(g):
            grad_buffer[idx] += grad

        # update state
        s = s_next

        if done:

            if render and i % 50 == 0:
                agent.env.render_fullcycle(state_history, action_history)

            # record how long it has been balancing when the simulation is done
            reward_itr += [current_reward]

            # apply gradients to the network variables
            feed_dict = dict(zip(agent.variable_pls, grad_buffer))
            sess.run(agent.update, feed_dict=feed_dict)

            # reset the buffer to zero
            for idx in range(len(grad_buffer)):
                grad_buffer[idx] *= 0
            break

    return state_history


def animate_itr(i, *args):
    #  animation of each training epoch
    agent, sess, grad_buffer, reward_itr, sess, grad_buffer, agent, obt_itr, render = args
    #
    state_history = one_trial(agent, sess, grad_buffer, reward_itr, i, render)
    xlist = [range(len(reward_itr))]
    ylist = [reward_itr]
    # if len(ylist) > 0:
    for lnum, line in enumerate(lines_itr):
        line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.
        # line.set_data(xlist[lnum], np.mean(ylist[np.max([lnum-100, 0]):lnum]))  # set data for each line separately.

    if len(reward_itr) % obt_itr == 0:
        xlist = np.asarray(state_history)[:, 0]
        ylist = np.asarray(state_history)[:, 1]
        lines_obt.set_data(xlist, ylist)
        lines_str.set_data(xlist[0], ylist[0])
        tau = 0.0167
        # time_text_obt.set_text('physical time = %6.2fs' % (len(xlist)*tau))
        time_text_obt.set_text('reward = %6.2f' % reward_itr[-1])

    return (lines_itr,) + (lines_obt,) + (lines_str,) + (time_text_obt,)


def get_fig(max_epoch):
    fig = plt.figure()
    ax_itr = axes([0.1, 0.1, 0.8, 0.8])
    ax_obt = axes([0.5, 0.2, .3, .3])

    # able to display multiple lines if needed
    global lines_obt, lines_itr, lines_str, time_text_obt
    lines_itr = []
    lobj = ax_itr.plot([], [], lw=1, color="blue")[0]
    lines_itr.append(lobj)
    lines_obt = []
    lines_str = []

    ax_itr.set_xlim([0, max_epoch])
    ax_itr.set_ylim([-300, 300]) # ([0, max_reward])
    ax_itr.grid(False)
    ax_itr.set_xlabel('training epoch')
    ax_itr.set_ylabel('reward')

    time_text_obt = []
    ax_obt.set_xlim([0, 1])
    ax_obt.set_ylim([1, 0])
    ax_obt.set_xlabel('x')
    ax_obt.set_ylabel('y')
    lines_obt = ax_obt.plot([], [], lw=1, color="red")[0]
    lines_str = ax_obt.plot([], [], linestyle='none', marker='o', color='g')[0]
    #plt.legend([lines_str], ['Start'])
    time_text_obt = ax_obt.text(0.05, 0.9, '', fontsize=13, transform=ax_obt.transAxes)
    return fig, ax_itr, ax_obt, time_text_obt


def main():
    obt_itr = 1
    max_epoch = 10000
    # whether to show the pole balancing animation
    render = True
    dir = 'tmp/trial/'

    # set up figure for animation
    fig, ax_itr, ax_obt, time_text_obt = get_fig(max_epoch)
    agent = Agent(hidden_size=12, lr=0.0001, gamma=0.95, dir=dir)
    agent.show_parameters()

    # tensorflow initialization for neural network controller
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    tf.global_variables_initializer().run(session=sess)
    grad_buffer = sess.run(tf.trainable_variables())
    tf.reset_default_graph()

    global reward_itr
    reward_itr = []
    args = [agent, sess, grad_buffer, reward_itr, sess, grad_buffer, agent, obt_itr, render]
    # run the optimization and output animation
    ani = animation.FuncAnimation(fig, animate_itr, fargs=args)
    plt.show()

if __name__ == "__main__":
   main()

# Set up formatting for the movie files
# print('saving animation...')
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)