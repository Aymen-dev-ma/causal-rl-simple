import datetime
from pathlib import Path

import mlflow
import whynot as wn
from causal_rl import ppo, vpg
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

# Set the tracking URI for mlflow to a local directory where you have write permissions
mlflow.set_tracking_uri("file:/Users/aymennasri/mlflow_tracking")

# Function to log parameters and metrics to mlflow with unique run_name
def log_to_mlflow(run_name, params, metrics):
    try:
        with mlflow.start_run(run_name=run_name):
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
    except Exception as e:
        print(f"Error logging to MLflow: {e}")

# Create the environment
env = wn.gym.make("HIV-v0")

# Create directories for logs and checkpoints
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
checkpoints_path = log_dir.joinpath("checkpoints")
checkpoints_path.mkdir(parents=True, exist_ok=True)

# Constants for training
EPOCHS = 20
EPISODES_PER_EPOCH = 16

# Training and logging VPG model
vpg_model = vpg.VPG(env)
for epoch in range(EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        # Perform training steps here
        # Replace with actual training logic
        train_stats = {"reward": 0}  # Example statistics; replace with actual values
        run_name = f"VPG_epoch_{epoch}_episode_{episode}"
        log_to_mlflow(run_name, {"epoch": epoch, "episode": episode}, train_stats)

# Saving VPG model
model_name = f"vpg_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
model_path = checkpoints_path.joinpath(model_name)
vpg_model.save(model_path)

# Log the model artifact to MLflow
with mlflow.start_run(run_name="vpg_model_artifact"):
    mlflow.log_artifact(model_path)

# Repeat similar process for other models (causal_pg, ppo, causal_ppo)...
# Example for ppo model
ppo_model = ppo.PPO(env)
for epoch in range(EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        # Perform training steps here
        # Replace with actual training logic
        train_stats = {"reward": 0}  # Example statistics; replace with actual values
        run_name = f"PPO_epoch_{epoch}_episode_{episode}"
        log_to_mlflow(run_name, {"epoch": epoch, "episode": episode}, train_stats)

# Saving PPO model
ppo_model_name = f"ppo_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
ppo_model_path = checkpoints_path.joinpath(ppo_model_name)
ppo_model.save(ppo_model_path)

# Log the PPO model artifact to MLflow
with mlflow.start_run(run_name="ppo_model_artifact"):
    mlflow.log_artifact(ppo_model_path)

# Example for plotting agent behaviors
agents = {
    "vpg": vpg_model,
    "ppo": ppo_model,
    # Add other models here
}

for name, agent in agents.items():
    plot_agent_behaviors(
        {name: agent},
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(
            f"{name}_alone_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
        ),
        show_plot=False,
    )
    plot_agent_behaviors(
        {
            "random": RandomPolicy(),
            "max": MaxTreatmentPolicy(),
            "none": NoTreatmentPolicy(),
            name: agent,
        },
        env,
        state_names=wn.hiv.State.variable_names(),
        max_timesteps=100,
        save_path=Path("logs").joinpath(
            f"{name}_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
        ),
        show_plot=False,
    )

# Example for plotting all agent behaviors together
plot_agent_behaviors(
    {
        "random": RandomPolicy(),
        "max": MaxTreatmentPolicy(),
        "none": NoTreatmentPolicy(),
        **agents,  # Include all trained agents
    },
    env,
    state_names=wn.hiv.State.variable_names(),
    max_timesteps=100,
    save_path=Path("logs").joinpath(
        f"all_behavior_{datetime.datetime.now():%d%m%y%H%M%S}.png"
    ),
    show_plot=False,
)
