import model.ActiveElastic as ae

n_agents = 3
n_steps = 10000
env_size = 100
graph_freq = 10
visualize = True
follow = True

sim = ae.SwarmSimulation(n_agents, n_steps, env_size, visualize, follow, graph_freq)
sim.run()