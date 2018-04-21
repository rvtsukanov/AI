from collections import namedtuple

import numpy as np
import sys

from gym import spaces

from env_core import CoreEnv


def up_scaler(grid, up_size):
    res = np.zeros(shape=np.asarray(np.shape(grid)) * up_size)
    for (x, y), value in np.ndenumerate(grid):
        res[x * up_size:x * up_size + up_size, y * up_size:y * up_size + up_size] = grid[x][y]
    return res


class Agent():
    def __init__(self, pos_x, pos_y, magnet_toogle=False):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.toogle = magnet_toogle


class ArmEnv(CoreEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    ACTIONS = namedtuple("ACTIONS", ["LEFT", "UP", "RIGHT", "DOWN", "ON", "OFF", ])(
        LEFT=0,
        UP=1,
        RIGHT=2,
        DOWN=3,
        ON=4,
        OFF=5,
    )

    MOVE_ACTIONS = {
        ACTIONS.UP: [-1, 0],
        ACTIONS.LEFT: [0, -1],
        ACTIONS.DOWN: [1, 0],
        ACTIONS.RIGHT: [0, 1],
    }

    def __init__(self, size_x, size_y, agents_num, cubes_cnt, episode_max_length,
                 finish_reward, action_minus_reward, tower_target_size):
        self._size_x = size_x
        self._size_y = size_y

        self._agents_num = agents_num
        self.agents = []

        self._cubes_cnt = cubes_cnt
        self._episode_max_length = episode_max_length
        self._finish_reward = finish_reward
        self._action_minus_reward = action_minus_reward
        self._tower_target_size = tower_target_size
        # checking for grid overflow
        assert cubes_cnt < size_x * size_y, "Cubes overflow the grid"
        assert self._agents_num <= size_y, "Too many agents"

        self.reset()

        self.action_space = spaces.Discrete(6)
        self.grid_to_id = {}

    def ok(self, x, y):
        return 0 <= x < self._grid.shape[0] and 0 <= y < self._grid.shape[1]

    def ok_and_empty(self, x, y):
        return self.ok(x, y) and self._grid[x][y] == 0

    def grid_to_bin(self):
        grid = np.array(self._grid, copy=True)
        for agent in self.agents:
            print(agent.toogle)
            if not agent.toogle:
                grid[agent.pos_x, agent.pos_y] = 2
            else:
                grid[agent.pos_x, agent.pos_y] = 3
        s = []
        for i in np.nditer(grid):
            s.append(int(i))
        return tuple(s)

    def get_tower_height(self):
        h = 0
        for j in range(self._grid.shape[1]):
            t = 0
            for i in np.arange(self._grid.shape[0] - 1, 0, -1):
                if self._grid[i, j] == 1 and self._grid[i - 1, j] == 0 and (i + 1 == self._grid.shape[0] or self._grid[i + 1, j] == 1):
                    t = self._grid.shape[0] - i
                    break
            if t > h:
                h = t
        return h

    def step(self, a, options=False):

        if not options:
            self._episode_length += 1

        for num_agent, agent in enumerate(self.agents):
            if a[num_agent] in self.MOVE_ACTIONS:
                cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
                cube_x, cube_y = agent.pos_x + cube_dx, agent.pos_y + cube_dy
                if agent.toogle and self.ok(cube_x, cube_y) and self._grid[cube_x][cube_y] == 1:
                    new_arm_x, new_arm_y = agent.pos_x + self.MOVE_ACTIONS[a[num_agent]][0], \
                                           agent.pos_x + self.MOVE_ACTIONS[a[num_agent]][1]
                    new_cube_x, new_cube_y = new_arm_x + cube_dx, new_arm_y + cube_dy
                    self._grid[cube_x][cube_y] = 0
                    if self.ok_and_empty(new_arm_x, new_arm_y) and self.ok_and_empty(new_cube_x, new_cube_y):
                        agent.pos_x, agent.pos_y = new_arm_x, new_arm_y #!!!!!!!!!!!!! HERE
                        self._grid[new_cube_x][new_cube_y] = 1
                    else:
                        self._grid[cube_x][cube_y] = 1
                else:
                    new_arm_x, new_arm_y = agent.pos_x + self.MOVE_ACTIONS[a[num_agent]][0], \
                                           agent.pos_y + self.MOVE_ACTIONS[a[num_agent]][1]
                    if self.ok_and_empty(new_arm_x, new_arm_y):
                        agent.pos_x, agent.pos_y = new_arm_x, new_arm_y
                    else:
                        # cant move, mb -reward
                        pass
            elif a[num_agent] == self.ACTIONS.ON:
                agent.toogle = True
            elif a[num_agent] == self.ACTIONS.OFF:
                cube_dx, cube_dy = self.MOVE_ACTIONS[self.ACTIONS.DOWN]
                cube_x, cube_y = agent.pos_x + cube_dx, agent.pos_y + cube_dy
                if self.ok(cube_x, cube_y) and self._grid[cube_x, cube_y] == 1 and agent.toogle:
                    new_cube_x, new_cube_y = cube_x + cube_dx, cube_y + cube_dy
                    while self.ok_and_empty(new_cube_x, new_cube_y):
                        new_cube_x, new_cube_y = new_cube_x + cube_dx, new_cube_y + cube_dy
                    new_cube_x, new_cube_y = new_cube_x - cube_dx, new_cube_y - cube_dy
                    self._grid[new_cube_x, new_cube_y], self._grid[cube_x, cube_y] = self._grid[cube_x, cube_y], self._grid[new_cube_x, new_cube_y]
                    agent.toogle = False

        observation = self.grid_to_bin()
        self._current_state = observation
        reward = self._action_minus_reward
        info = None
        # self.render_to_image()
        # observation (object): agent's observation of the current environment
        # reward (float) : amount of reward returned after previous action
        # done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        # info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        if self.get_tower_height() == self._tower_target_size and self._magnet_toggle == False:
            self._done = True
            reward += self._finish_reward
            info = True
            return observation, reward, self._done, info

        if self._episode_max_length <= self._episode_length:
            self._done = True
        return observation, reward, self._done, info

    def is_done(self):
        return self._done

    # return observation
    def _get_obs(self):
        pass

    # return: (states, observations)
    def reset(self):
        self._episode_length = 0
        self._grid = np.zeros(shape=(self._size_x, self._size_y), dtype=np.int32)
        for i in range(self._agents_num):
            self.agents.append(Agent(0, i, False))
        self._done = False

        cubes_left = self._cubes_cnt
        for (x, y), value in reversed(list(np.ndenumerate(self._grid))):
            if cubes_left == 0:
                break
            cubes_left -= 1
            self._grid[x, y] = 1

        self._current_state = self.grid_to_bin()
        return self._current_state

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout

        out = np.array(self._grid, copy=True)
        for agent in self.agents:
            out[agent.pos_x, agent.pos_y] = 2 + agent.toogle * 1

        outfile.write('\n')
        outfile.write(str(out))
        outfile.write('\n')


env = ArmEnv(size_x=4,
             size_y=3,
             agents_num=2,
             cubes_cnt=4,
             episode_max_length=200,
             finish_reward=200,
             action_minus_reward=0.0,
             tower_target_size=3)

env.step((2, 1))
env.render()