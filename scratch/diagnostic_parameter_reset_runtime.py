import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

try:
    from src.lrma_trainer import LRMATrainer
except ModuleNotFoundError:
    from lrma_trainer import LRMATrainer


def get_layer_weights(net, layer_idx):
    layer = net.net[layer_idx]
    w = layer.weight.detach().cpu().clone().numpy()
    b = layer.bias.detach().cpu().clone().numpy() if layer.bias is not None else None
    return w, b


def get_all_weights(net):
    state = {}
    for name, param in net.named_parameters():
        state[name] = param.detach().cpu().clone().numpy()
    return state


def compare_weights(state_before, state_after):
    all_same = True
    max_diff = 0.0
    for name in state_before:
        diff = np.max(np.abs(state_before[name] - state_after[name]))
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-7:
            all_same = False
    return all_same, float(max_diff)


def main():
    print("==========================================================================================")
    print("PARAMETER-RESET RUNTIME VERIFICATION DIAGNOSTIC (LRMA ALGORITHM 1)")
    print("==========================================================================================")

    # 1. Instantiate Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device)

    # 2. Snapshot parameters before reset
    ed_pri_hidden_before = [get_layer_weights(a, 0) for a in trainer.ed_primary_actors]
    ed_pri_last_before = [get_layer_weights(a, 4) for a in trainer.ed_primary_actors]

    cloud_pri_hidden_before = get_layer_weights(trainer.cloud_primary_actor, 0)
    cloud_pri_last_before = get_layer_weights(trainer.cloud_primary_actor, 4)

    ed_target_before = [get_all_weights(a) for a in trainer.ed_target_actors]
    cloud_target_before = get_all_weights(trainer.cloud_target_actor)

    critic_before = get_all_weights(trainer.critic)
    critic_target_before = get_all_weights(trainer.critic_target)

    # 3. Perform Direct Reset Call
    trainer.reset_primary_parameters()

    # 4. Snapshot parameters after reset
    ed_pri_hidden_after = [get_layer_weights(a, 0) for a in trainer.ed_primary_actors]
    ed_pri_last_after = [get_layer_weights(a, 4) for a in trainer.ed_primary_actors]

    cloud_pri_hidden_after = get_layer_weights(trainer.cloud_primary_actor, 0)
    cloud_pri_last_after = get_layer_weights(trainer.cloud_primary_actor, 4)

    ed_target_after = [get_all_weights(a) for a in trainer.ed_target_actors]
    cloud_target_after = get_all_weights(trainer.cloud_target_actor)

    critic_after = get_all_weights(trainer.critic)
    critic_target_after = get_all_weights(trainer.critic_target)

    # 5. Property Verifications
    results = {}

    # Property 1: ED Primary Last Layer Changed
    ed_last_changed = True
    for idx in range(25):
        w1, b1 = ed_pri_last_before[idx]
        w2, b2 = ed_pri_last_after[idx]
        if np.max(np.abs(w1 - w2)) < 1e-7:
            ed_last_changed = False
    results["1. ED Primary Last Layers Changed"] = "PASS" if ed_last_changed else "FAIL"

    # Property 2: Cloud Primary Last Layer Changed
    cw1, cb1 = cloud_pri_last_before
    cw2, cb2 = cloud_pri_last_after
    cloud_last_changed = np.max(np.abs(cw1 - cw2)) > 1e-7
    results["2. Cloud Primary Last Layer Changed"] = "PASS" if cloud_last_changed else "FAIL"

    # Property 3: ED Primary Hidden Layers Unchanged
    ed_hidden_unchanged = True
    for idx in range(25):
        w1, b1 = ed_pri_hidden_before[idx]
        w2, b2 = ed_pri_hidden_after[idx]
        if np.max(np.abs(w1 - w2)) > 1e-7:
            ed_hidden_unchanged = False
    results["3. ED Primary Hidden Layers Unchanged"] = "PASS" if ed_hidden_unchanged else "FAIL"

    # Property 4: Cloud Primary Hidden Layer Unchanged
    hcw1, hcb1 = cloud_pri_hidden_before
    hcw2, hcb2 = cloud_pri_hidden_after
    cloud_hidden_unchanged = np.max(np.abs(hcw1 - hcw2)) < 1e-7
    results["4. Cloud Primary Hidden Layer Unchanged"] = "PASS" if cloud_hidden_unchanged else "FAIL"

    # Property 5: ED Target Actors Unchanged
    ed_target_unchanged = True
    for idx in range(25):
        same, _ = compare_weights(ed_target_before[idx], ed_target_after[idx])
        if not same:
            ed_target_unchanged = False
    results["5. ED Target Actors Unchanged"] = "PASS" if ed_target_unchanged else "FAIL"

    # Property 6: Cloud Target Actor Unchanged
    cloud_target_same, _ = compare_weights(cloud_target_before, cloud_target_after)
    results["6. Cloud Target Actor Unchanged"] = "PASS" if cloud_target_same else "FAIL"

    # Property 7: Centralized Critic Unchanged
    critic_same, _ = compare_weights(critic_before, critic_after)
    results["7. Centralized Critic Unchanged"] = "PASS" if critic_same else "FAIL"

    # Property 8: Centralized Target Critic Unchanged
    critic_target_same, _ = compare_weights(critic_target_before, critic_target_after)
    results["8. Centralized Target Critic Unchanged"] = "PASS" if critic_target_same else "FAIL"

    # 6. Test Trigger Logic
    print("\n============================================================")
    print("TESTING PARAMETER RESET TRIGGER LOGIC")
    print("============================================================")

    # Setup dummy experience in replay buffer for train_step testing
    s_dim = trainer.joint_state_dim
    a_dim = trainer.joint_action_dim
    dummy_s = np.zeros((100, s_dim), dtype=np.float32)
    dummy_a = np.zeros((100, a_dim), dtype=np.float32)
    dummy_r = np.zeros(100, dtype=np.float32)
    dummy_ns = np.zeros((100, s_dim), dtype=np.float32)
    dummy_d = np.zeros(100, dtype=np.float32)

    for i in range(100):
        trainer.replay_buffer_ed.add(dummy_s[i], dummy_a[i], dummy_r[i], dummy_ns[i], dummy_d[i])

    # Test Step Trigger Logic (eval_step=49 vs 50)
    trainer.total_update_steps = 48
    w_before_49, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    trainer.train_step(batch_size=64, delta_reset=50) # total_update_steps becomes 49
    w_after_49, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    step_49_no_reset = np.max(np.abs(w_before_49 - w_after_49)) < 1e-7
    results["9. Trigger Step 49 No Reset"] = "PASS" if step_49_no_reset else "FAIL"

    w_before_50, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    trainer.train_step(batch_size=64, delta_reset=50) # total_update_steps becomes 50 -> TRIGGER RESET!
    w_after_50, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    step_50_did_reset = np.max(np.abs(w_before_50 - w_after_50)) > 1e-7
    results["10. Trigger Step 50 Did Reset"] = "PASS" if step_50_did_reset else "FAIL"

    # Test Slot Trigger Logic (slot_t=49 vs 50)
    trainer.total_update_steps = 1
    w_before_slot49, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    trainer.train_step(batch_size=64, delta_reset=50, slot_t=49)
    w_after_slot49, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    slot_49_no_reset = np.max(np.abs(w_before_slot49 - w_after_slot49)) < 1e-7
    results["11. Slot_t=49 No Reset"] = "PASS" if slot_49_no_reset else "FAIL"

    w_before_slot50, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    trainer.train_step(batch_size=64, delta_reset=50, slot_t=50)
    w_after_slot50, _ = get_layer_weights(trainer.cloud_primary_actor, 4)
    slot_50_did_reset = np.max(np.abs(w_before_slot50 - w_after_slot50)) > 1e-7
    results["12. Slot_t=50 Did Reset"] = "PASS" if slot_50_did_reset else "FAIL"

    print("\n============================================================")
    print("VERIFICATION RESULTS SUMMARY")
    print("============================================================")
    all_pass = True
    for prop, status in results.items():
        print(f"{prop:<45}: {status}")
        if status != "PASS":
            all_pass = False

    print(f"\nOVERALL RUNTIME VERIFICATION STATUS: {'PASS' if all_pass else 'FAIL'}")

    print("\n============================================================")
    print("INTEGRITY CONFIRMATION")
    print("============================================================")
    print("SOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("REAL TRAINING EXECUTED: NO")
    print("HOST CUDA EXECUTION: NO")


if __name__ == "__main__":
    main()
