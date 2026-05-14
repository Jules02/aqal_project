import json
import numpy as np
from scipy import stats

# Read data
with open("../experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

quantum_train_acc   = [r["final_train_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_test_acc   = [r["final_test_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_novel_test_acc   = [r["novel_test_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_total_time  = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
classical_train_acc = [r["final_train_acc"] for r in records if r["experiment_name"] == "non_equiv_quantum_model"]
classical_test_acc = [r["final_test_acc"] for r in records if r["experiment_name"] == "non_equiv_quantum_model"]
classical_novel_test_acc = [r["novel_test_acc"] for r in records if r["experiment_name"] == "non_equiv_quantum_model"]
classical_total_time = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "non_equiv_quantum_model"]




X_train_acc = np.array(quantum_train_acc)
Y_train_acc = np.array(classical_train_acc)
print(f"Mean equiv train acc   : {X_train_acc.mean():.4f} ± {X_train_acc.std():.4f}")
print(f"Mean non-equiv train acc : {Y_train_acc.mean():.4f} ± {Y_train_acc.std():.4f}")
print()

X_test_acc = np.array(quantum_test_acc)
Y_test_acc = np.array(classical_test_acc)
print(f"Mean equiv test acc   : {X_test_acc.mean():.4f} ± {X_test_acc.std():.4f}")
print(f"Mean non-equiv test acc : {Y_test_acc.mean():.4f} ± {Y_test_acc.std():.4f}")
print()

X_novel_test_acc = np.array(quantum_novel_test_acc)
Y_novel_test_acc = np.array(classical_novel_test_acc)
print(f"Mean equiv novel test acc   : {X_novel_test_acc.mean():.4f} ± {X_novel_test_acc.std():.4f}")
print(f"Mean non-equiv novel test acc : {Y_novel_test_acc.mean():.4f} ± {Y_novel_test_acc.std():.4f}")
print()

X_total_time = np.array(quantum_total_time)
Y_total_time = np.array(classical_total_time)
print(f"Mean equiv total_time   : {X_total_time.mean():.4f} ± {X_total_time.std():.4f}")
print(f"Mean non-equiv total_time : {Y_total_time.mean():.4f} ± {Y_total_time.std():.4f}")
