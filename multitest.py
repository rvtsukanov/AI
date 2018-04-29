from trpo_class_multi import MultiAC
from arm_env_multi import ArmEnv

env = ArmEnv(size_x=4,
             size_y=3,
             agents_num=2,
             cubes_cnt=4,
             episode_max_length=200,
             finish_reward=200,
             action_minus_reward=0.0,
             tower_target_size=3)

ac = MultiAC(env=env, num_episodes=1000)
ac.learn()