import numpy as np
import pygame
from slimevolleygym import SlimeVolleyEnv
from slimevolleygym.slimevolley import setPixelObsMode

from agents import HeuristicPolicy, OptimizedPolicy, RLAgent


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["human", "heuristic", "rl", "opt"],
        default="human",
        help="Choose who controls the right slime.",
    )
    parser.add_argument(
        "--opponent",
        choices=["default", "heuristic", "rl", "opt"],
        default="default",
        help="Choose who controls the left slime (opponent).",
    )
    return parser.parse_args()


TARGET_FPS = 60
WINDOW_SCALE = 3


def read_action():
    keys = pygame.key.get_pressed()
    action = [0, 0, 0]

    if keys[pygame.K_a]:
        action[0] = 1
    if keys[pygame.K_d]:
        action[1] = 1
    if keys[pygame.K_w]:
        action[2] = 1

    return action


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            return False
    return True


setPixelObsMode()
args = parse_args()
env = SlimeVolleyEnv()
env.seed(np.random.randint(0, 10000))
obs = env.reset()
if args.agent == "heuristic":
    agent = HeuristicPolicy()
elif args.agent == "rl":
    agent = RLAgent()
elif args.agent == "opt":
    agent = OptimizedPolicy()
else:
    agent = None

if args.opponent == "heuristic":
    env.policy = HeuristicPolicy()
elif args.opponent == "rl":
    env.policy = RLAgent()
elif args.opponent == "opt":
    env.policy = OptimizedPolicy()

first_frame = env.render(mode="state")
height, width, _ = first_frame.shape
window_size = (width * WINDOW_SCALE, height * WINDOW_SCALE)

pygame.init()
window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Slime Valley")
clock = pygame.time.Clock()

running = True
while running:
    running = handle_events()
    if not running:
        break

    frame = env.render(mode="state")
    # Pygame expects pixels in (width, height, channels) ordering.
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if WINDOW_SCALE != 1:
        surface = pygame.transform.scale(surface, window_size)
    window.blit(surface, (0, 0))
    pygame.display.flip()

    action = agent.predict(obs) if agent else read_action()
    obs, reward, done, _info = env.step(action)

    if reward:
        print(f"Point: {reward:+.0f}")

    if done:
        obs = env.reset()
        if agent:
            agent.reset()

    clock.tick(TARGET_FPS)

env.close()
pygame.quit()
