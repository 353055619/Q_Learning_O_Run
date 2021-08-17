from src.Q_Learning_O_Run.env import Env
from src.Q_Learning_O_Run.Q_learning import build_Q_Table,Brain


if __name__ == '__main__':
    env = Env(length=6, init_postion=0, fresh_time=0.3)
    q_table = build_Q_Table(Env.ACTION, Size=6)
    q_brain = Brain(q_table, Env.ACTION, Epsilon=0.9, Alpha=0.1, Gamma=0.9)
    for episode in range(30):
        env.refresh()
        print(q_table)
        step, step_, done = 0, 0, False
        counter = 0
        while not done:
            action = q_brain.choose_Action(step)
            reward, step_, done = env.update_env(action)
            q_brain.update_Q_Table(action, step, reward, step_, done)
            counter += 1
            print('Episode:{}, total steps:{}'.format(episode + 1, counter))