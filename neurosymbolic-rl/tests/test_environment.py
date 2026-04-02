def test_episode_completion():
    env = SymbolicExpressionEnv(...)
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
    assert reward is not None