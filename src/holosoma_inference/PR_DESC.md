# Step 8 — what it eliminates

Things that simply stop existing:

- `DualModePolicy` class
- `BasePolicy._handle_start_policy` / `_handle_stop_policy` / `_handle_init_state` / `_handle_damp_state`
- `use_policy_action`, `get_ready_state`, `_stiff_hold_active`, `init_count` flags
- `Controller.state` enum + `set_state` writethrough
- `_publish_damp_command` (lives on `DampingPolicy.act` instead)
- `policy_action()`'s 5-way branching
- `ControllerState` enum (replaced by `controller.active.name`)
- `_shared_hardware_source` (already gone) and any remnants of guard patterns
- The `_dispatch_command` lambda-patching in `DualModePolicy.bind_controller`
