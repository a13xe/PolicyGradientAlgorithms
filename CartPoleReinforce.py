import random
import gym

env = gym.make("CartPole-v1", render_mode="human")

episodes = 15
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        action = random.choice([0,1])
        n_step, reward, done, info = env.step(action)
        score += reward
        env.render()
        
    print(f"Episode {episode}, Score: {score}")
        
        