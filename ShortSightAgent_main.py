from agents.short_sight_agent import ShortSightAgent
from training.short_sight_agent_reinforce import run_game, transform_sample
import numpy as np

if __name__ == "__main__":

    agent = ShortSightAgent(greedy=False, learning_rate=0.001)
    #agent.load_weights('ShortSightAgent')
    for iteration in range(1, 201):
        samples = []
        avg_duration = []
        nb_games = 100 * (1+int(float(iteration)/10))
        nb_games = min(nb_games, 1000)
        print(f'# games to play: {nb_games}')
        for _ in range(nb_games):
            cur_game = run_game(np.random.randint(7)+1, agent)
            samples += cur_game
            avg_duration.append(len(cur_game))
        avg_nb_steps = np.mean(avg_duration)
        max_nb_steps = np.max(avg_duration)
        print(f'Average game duration (steps): {avg_nb_steps}')
        print(f'Max game duration (steps): {max_nb_steps}')
        X, y, X_val, y_val = transform_sample(samples)
        agent.fit(X, y, X_val, y_val)
        if (iteration % 10) == 0:
            print('Saving model weights')
            agent.save_weights('ShortSightAgent')
