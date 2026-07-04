# Raw policy output this tick (pre-scale network action). header.stamp matches
# the paired /holosoma/observation so consumers can zip (obs -> action) pairs.
std_msgs/Header header
string wandb_id
float32[] data
