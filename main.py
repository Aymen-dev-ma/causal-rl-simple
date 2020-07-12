from pathlib import Path

import whynot as wn

from causal_rl import ppo, vpg  # noqa: F401
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors

env = wn.gym.make("HIV-v0")
# print(wn.hiv.State.variable_names())

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
checkpoints_path = log_dir.joinpath("checkpoints")
checkpoints_path.mkdir(parents=True, exist_ok=True)

EPOCHS = 1
EPISODES_PER_EPOCH = 1

# vpg_model = vpg.VPG(env)
# vpg_model.train(
#     epochs=EPOCHS,
#     episodes_per_epoch=EPISODES_PER_EPOCH,
#     # value_lr=VALUE_FN_LEARNING_RATE,
#     # policy_lr=POLICY_LEARNING_RATE,
#     # gamma=GAMMA,
#     # max_traj_length=MAX_TRAJ_LENGTH,
#     log_dir=log_dir,
#     # RENDER=False,
#     PLOT_REWARDS=True,
#     VERBOSE=True,
#     TENSORBOARD_LOG=True,
# )
# vpg_model.save(checkpoints_path.joinpath("vpg_model.pt"))

# causal_pg_model = vpg.CausalPG(compute_causal_factor1, env)
# causal_pg_model.train(
#     epochs=2,
#     episodes_per_epoch=16,
#     # value_lr=VALUE_FN_LEARNING_RATE,
#     # policy_lr=POLICY_LEARNING_RATE,
#     # gamma=GAMMA,
#     # max_traj_length=MAX_TRAJ_LENGTH,s
#     log_dir=log_dir,
#     # RENDER=False,
#     PLOT_REWARDS=True,
#     VERBOSE=True,
#     TENSORBOARD_LOG=True,)
# causal_pg_model.save(checkpoints_path.joinpath("causal_pg_model.pt"))

# ppo_model = ppo.PPO(env)
# ppo_model.train(
#     epochs=EPOCHS,
#     episodes_per_epoch=EPISODES_PER_EPOCH,
#     # n_value_updates=N_VALUE_UPDATES,
#     # n_policy_updates=N_POLICY_UPDATES,
#     # value_lr=VALUE_FN_LEARNING_RATE,
#     # policy_lr=POLICY_LEARNING_RATE,
#     # gamma=GAMMA,
#     # epsilon=EPSILON,
#     # max_traj_length=MAX_TRAJ_LENGTH,
#     log_dir=log_dir,
#     # RENDER=False,
#     PLOT_REWARDS=True,
#     VERBOSE=True,
#     TENSORBOARD_LOG=True,
# )
# ppo_model.save(checkpoints_path.joinpath("ppo_model.pt"))

causal_ppo_model = ppo.CausalPPO(compute_causal_factor1, env)
# causal_ppo_model.train(
#     epochs=EPOCHS,
#     episodes_per_epoch=EPISODES_PER_EPOCH,
#     # n_value_updates=N_VALUE_UPDATES,
#     # n_policy_updates=N_POLICY_UPDATES,
#     # value_lr=VALUE_FN_LEARNING_RATE,
#     # policy_lr=POLICY_LEARNING_RATE,
#     # gamma=GAMMA,
#     # epsilon=EPSILON,
#     # max_traj_length=MAX_TRAJ_LENGTH,
#     log_dir=log_dir,
#     # RENDER=False,
#     PLOT_REWARDS=True,
#     VERBOSE=True,
#     TENSORBOARD_LOG=True,
# )
# causal_ppo_model.save(checkpoints_path.joinpath("causal_pg_model.pt"))

model_path = Path("logs/12072020165617/CausalPPO_HIV-v0.pt")
causal_ppo_model.load(model_path)
plot_agent_behaviors(
    {"causal_ppo": causal_ppo_model},
    env,
    state_names=wn.hiv.State.variable_names(),
    max_timesteps=100,
    save_path=model_path.parent.joinpath("behavior.png")
)
