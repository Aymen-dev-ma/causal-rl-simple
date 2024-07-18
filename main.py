import datetime
from pathlib import Path

import mlflow
import whynot as wn
from causal_rl import ppo, vpg
from causal_rl.common import compute_causal_factor1, plot_agent_behaviors
from causal_rl.common import NoTreatmentPolicy, RandomPolicy, MaxTreatmentPolicy

# Set the tracking URI for MLflow to a local directory where you have write permissions
mlflow.set_tracking_uri("file:/Users/aymennasri/mlflow_tracking")

# Function to log parameters and metrics to MLflow with unique run_name
def log_to_mlflow(run_name, params, metrics):
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

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
        train_stats = {"reward": 0.0}  # Example statistics; replace with actual values
        run_name = f"VPG_epoch_{epoch}_episode_{episode}"
        
        # Log parameters and metrics to MLflow
        log_to_mlflow(run_name, {"epoch": epoch, "episode": episode}, train_stats)

# Save the VPG model
model_name = f"vpg_model_{datetime.datetime.now():%d%m%y%H%M%S}.pt"
vpg_model.save(checkpoints_path.joinpath(model_name))

# Example for plotting agent behaviors
agents = {
    "vpg": vpg_model,
    # Add other models here if applicable
}

for name, agent in agents.items():
    # Plot and save individual agent behavior
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

# Log final model save path as an artifact
with mlflow.start_run(run_name="final_model"):
    mlflow.log_artifact(checkpoints_path.joinpath(model_name))
