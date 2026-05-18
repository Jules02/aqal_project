import json
from pathlib import Path
import numpy as np
from scipy import stats

# Read data
log_path = Path(__file__).resolve().parent.parent / "experiments_log.jsonl"
with open(log_path) as f:
    records = [json.loads(line) for line in f if line.strip()]

quantum_train_acc   = [r["final_train_acc"] for r in records if r["experiment_name"] == "quantum_v_classical"]
quantum_test_acc   = [r["final_test_acc"] for r in records if r["experiment_name"] == "quantum_v_classical"]
quantum_novel_test_acc = [r["novel_test_acc"] for r in records if r["experiment_name"] == "quantum_v_classical"]
quantum_total_time  = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "quantum_v_classical"]
classical_train_acc = [r["final_train_acc"] for r in records if r["experiment_name"] == "classical_baseline"]
classical_test_acc = [r["final_test_acc"] for r in records if r["experiment_name"] == "classical_baseline"]
classical_novel_test_acc = [r["novel_test_acc"] for r in records if r["experiment_name"] == "classical_baseline"]
classical_total_time = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "classical_baseline"]

print(len(quantum_train_acc))
print(len(classical_train_acc))

X_train_acc = np.array(quantum_train_acc)
Y_train_acc = np.array(classical_train_acc)
print(f"Mean quantum train acc   : {X_train_acc.mean():.3f} ± {X_train_acc.std():.3f}")
print(f"Mean classical train acc : {Y_train_acc.mean():.3f} ± {Y_train_acc.std():.3f}")
print()

X_test_acc = np.array(quantum_test_acc)
Y_test_acc = np.array(classical_test_acc)
print(f"Mean quantum test acc   : {X_test_acc.mean():.3f} ± {X_test_acc.std():.3f}")
print(f"Mean classical test acc : {Y_test_acc.mean():.3f} ± {Y_test_acc.std():.3f}")
print()

X_novel_test_acc = np.array(quantum_novel_test_acc)
Y_novel_test_acc = np.array(classical_novel_test_acc)
print(f"Mean quantum novel test acc   : {X_novel_test_acc.mean():.3f} ± {X_novel_test_acc.std():.3f}")
print(f"Mean classical novel test acc : {Y_novel_test_acc.mean():.3f} ± {Y_novel_test_acc.std():.3f}")
print()

X_total_time = np.array(quantum_total_time)
Y_total_time = np.array(classical_total_time)
print(f"Mean quantum total_time   : {X_total_time.mean():.3f} ± {X_total_time.std():.3f}")
print(f"Mean classical total_time : {Y_total_time.mean():.3f} ± {Y_total_time.std():.3f}")
