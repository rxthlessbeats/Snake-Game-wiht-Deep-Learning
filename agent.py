import torch
import random
import numpy as np
from collections import deque
from snake_ai import SnakeGame_AI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot 

MAX_MEMORY = 1_000_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #random
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        direction_l = game.direction == Direction.LEFT
        direction_r = game.direction == Direction.RIGHT
        direction_u = game.direction == Direction.UP
        direction_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (direction_r and game.collision(point_r)) or
            (direction_l and game.collision(point_l)) or
            (direction_u and game.collision(point_u)) or
            (direction_d and game.collision(point_d)), 

            #danger right
            (direction_u and game.collision(point_r)) or
            (direction_r and game.collision(point_d)) or
            (direction_d and game.collision(point_l)) or
            (direction_l and game.collision(point_u)), 

            #danger left
            (direction_u and game.collision(point_l)) or
            (direction_l and game.collision(point_d)) or
            (direction_d and game.collision(point_r)) or
            (direction_r and game.collision(point_u)),

            #move direction
            direction_d, direction_l, direction_u, direction_r,

            #food direction
            game.food.x < game.head.x, #right
            game.food.x > game.head.x, #left
            game.food.y < game.head.y, #up 
            game.food.y > game.head.y  #down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)   
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
                
def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = SnakeGame_AI()

    while True:
        #get old state
        old_state = agent.get_state(game)
        #final move
        final_move = agent.get_action(old_state)
        #perform move
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)
        #train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        #remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            #train long memory and plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Round:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()