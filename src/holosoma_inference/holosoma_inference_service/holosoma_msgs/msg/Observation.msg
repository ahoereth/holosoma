# Raw policy observation vector fed to the network this tick (actor_obs).
# Slice-by-name layout is described once on /holosoma/policy_metadata; wandb_id
# links this sample to the metadata that describes it (survives policy swaps).
std_msgs/Header header
string wandb_id
float32[] data
