import torch
class CMTR_Visualization: 
    def __init__(self, env, env_config, num_envs, model): 
        self.env = env
        self.env_config = env_config
        self.num_envs = num_envs
        self.model = model

    def get_frames(self): 
        obs = self.env.reset()
        frames = {f"env_{i}": [] for i in range(self.num_envs)}

        for t in range(self.env_config.episode_len):

            obs = torch.reshape(obs, (-1, obs.shape[-1]))
            
            # Sample random actions
            action = policy(obs, True)[0]

            action = torch.reshape(action, (self.num_envs, MAX_NUM_OBJECTS, -1))

            # Step the environment
            env.step_dynamics(action)

            obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()

            # Render the environment    
            if t % 5 == 0:
                imgs = env.vis.plot_simulator_state(
                    env_indices=list(range(self.num_envs)),
                    time_steps=[t]*self.num_envs,
                    zoom_radius=70,
                )
            
                for i in range(self.num_envs):
                    frames[f"env_{i}"].append(img_from_fig(imgs[i])) 
                
            if done.all():
                break
        return frames