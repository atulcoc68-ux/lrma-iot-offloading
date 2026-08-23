import sys
import os

def main():
    print("==========================================================================================")
    print("ACTOR UPDATE EQUATION FIDELITY AUDIT REPORT (IEEE TNSE 2025 vs CURRENT CODEBASE)")
    print("==========================================================================================")

    audit_items = [
        {
            "id": "A",
            "component": "Candidate Action Generation (P candidates)",
            "paper_ref": "Section V-B.2, Eq. (40)-(41)",
            "file": "src/lrma_candidates.py",
            "func": "point_to_uniform_candidate_generation",
            "operation": "Categorical sampling of P=5 candidate actions from actor probability dist.",
            "status": "PASS",
            "reason": "Correctly generates P=5 candidate solutions during environment execution step."
        },
        {
            "id": "B",
            "component": "Candidate Action Evaluation",
            "paper_ref": "Section V-B.2, Eq. (42)",
            "file": "src/lrma_candidates.py",
            "func": "select_best_candidate",
            "operation": "Evaluates expected utility for each candidate and picks argmax.",
            "status": "PASS",
            "reason": "Evaluates scalar utility for candidate actions during trajectory rollout."
        },
        {
            "id": "C",
            "component": "Candidate Impact on Actor Update",
            "paper_ref": "Section V-B.2, Eq. (43)-(44)",
            "file": "src/lrma_trainer.py",
            "func": "train_step (L164-L183)",
            "operation": "Bypasses candidates during training step; passes continuous softmax probs to critic.",
            "status": "MISMATCH",
            "reason": "Candidate selection is used only for environment step, NOT inside actor parameter gradient step."
        },
        {
            "id": "D",
            "component": "Actor Objective Formulation",
            "paper_ref": "Section V-B.2, Eq. (43)-(44)",
            "file": "src/lrma_trainer.py",
            "func": "train_step (L170)",
            "operation": "actor_loss = -critic(states, joint_a_diff).mean()",
            "status": "MISMATCH",
            "reason": "Uses continuous DDPG deterministic policy gradient -Q(S, a_i + replay_others) instead of candidate expectation."
        },
        {
            "id": "E",
            "component": "Actor Loss Equivalence (-Q(S, a_i + replay_others))",
            "paper_ref": "Section V-B.2 / Section V-A",
            "file": "src/lrma_trainer.py",
            "func": "train_step (L170 & L180)",
            "operation": "Substitutes actor probability vector into replay joint action.",
            "status": "PARTIAL",
            "reason": "Continuous relaxation of discrete MARL action space; mathematically convenient for Autograd but differs from paper discrete candidate expectation."
        },
        {
            "id": "F",
            "component": "Replayed Actions of Other Agents Usage",
            "paper_ref": "Section V-A / Algorithm 1",
            "file": "src/lrma_trainer.py",
            "func": "train_step (L167-L168)",
            "operation": "joint_a_diff[:, idx*2:(idx+1)*2] = pred_probs",
            "status": "PASS",
            "reason": "Faithfully holds actions of other N-1 agents fixed from replay buffer during single-agent actor update."
        },
        {
            "id": "G",
            "component": "Candidate Evaluation During Training",
            "paper_ref": "Section V-B.2",
            "file": "src/lrma_trainer.py",
            "func": "train_step",
            "operation": "Direct critic evaluation on continuous probability distribution.",
            "status": "MISMATCH",
            "reason": "Training step bypasses candidate evaluation mechanism completely."
        },
        {
            "id": "H",
            "component": "Action Representation Alignment",
            "paper_ref": "Section III-B, Eq. (5)-(7)",
            "file": "src/lrma_trainer.py & src/lrma_candidates.py",
            "func": "select_ed_action vs train_step",
            "operation": "Discrete integers {0, 1} for rollout vs 2-D continuous [P(local), P(offload)] for gradient.",
            "status": "PARTIAL",
            "reason": "Candidate selection uses discrete indices; actor gradient pass uses continuous probability relaxation."
        },
        {
            "id": "I",
            "component": "Gradient Flow Target",
            "paper_ref": "Section V-B.2",
            "file": "src/lrma_trainer.py",
            "func": "train_step (L171)",
            "operation": "actor_loss.backward() through CentralizedCritic to Actor parameters.",
            "status": "PASS",
            "reason": "Correct end-to-end autograd backpropagation from centralized critic into actor parameters."
        }
    ]

    header_fmt = "{:<3} | {:<42} | {:<12} | {:<25} | {:<12}"
    print(header_fmt.format("ID", "Audit Item", "Paper Ref", "Code Location", "Status"))
    print("-" * 105)

    status_counts = {}
    for item in audit_items:
        st = item["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        print(header_fmt.format(item["id"], item["component"], item["paper_ref"], item["file"], item["status"]))

    print("\n==========================================================================================")
    print("DETAILED AUDIT DISCREPANCY FINDINGS")
    print("==========================================================================================")
    for item in audit_items:
        print(f"[{item['id']}] {item['component']}")
        print(f"    Paper Reference : {item['paper_ref']}")
        print(f"    Code Location   : {item['file']} -> {item['func']}")
        print(f"    Current Op      : {item['operation']}")
        print(f"    Status          : {item['status']}")
        print(f"    Reason          : {item['reason']}\n")

    print("==========================================================================================")
    print("FINAL CLASSIFICATION")
    print("==========================================================================================")

    if status_counts.get("MISMATCH", 0) > 0:
        classification = "B. ACTOR UPDATE USES CONTINUOUS DDPG RELAXATION RATHER THAN DISCRETE CANDIDATE OBJECTIVE"
    elif status_counts.get("PASS", 0) == len(audit_items):
        classification = "A. ACTOR UPDATE FULLY MATCHES PAPER SPECIFICATION"
    else:
        classification = "C. ACTOR UPDATE HAS STRUCTURAL DISCREPANCIES"

    print(f"CLASSIFICATION: {classification}")

    print("\nStatus Breakdown:")
    for st, count in status_counts.items():
        print(f"  - {st:<15}: {count}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("REAL TRAINING EXECUTED: NO")
    print("HOST CUDA EXECUTION: NO")


if __name__ == "__main__":
    main()
