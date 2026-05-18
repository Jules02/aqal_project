import json
import numpy as np
from scipy import stats

# Read data
with open("experiments_log.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

quantum_train_acc   = [r["final_train_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_test_acc   = [r["final_test_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_novel_test_acc   = [r["novel_test_acc"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_total_time  = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
quantum_avg_time = [r["avg_time_per_epoch_sec"] for r in records if r["experiment_name"] == "sec2" and r["num_layers"] == 2 and r["num_params_per_layer"] == 2]
non_equiv_train_acc = [r["final_train_acc"] for r in records if r["experiment_name"] == "improved_ansatz" and r["num_layers"] == 2 and r["num_params_per_layer"] == 3]
non_equiv_test_acc = [r["final_test_acc"] for r in records if r["experiment_name"] == "improved_ansatz" and r["num_layers"] == 2 and r["num_params_per_layer"] == 3]
non_equiv_novel_test_acc = [r["novel_test_acc"] for r in records if r["experiment_name"] == "improved_ansatz" and r["num_layers"] == 2 and r["num_params_per_layer"] == 3]
non_equiv_total_time = [r["total_training_time_sec"] for r in records if r["experiment_name"] == "improved_ansatz" and r["num_layers"] == 2 and r["num_params_per_layer"] == 3]
non_equiv_avg_time = [r["avg_time_per_epoch_sec"] for r in records if r["experiment_name"] == "improved_ansatz" and r["num_layers"] == 2 and r["num_params_per_layer"] == 3]

print(len(non_equiv_train_acc))
print(len(quantum_train_acc))


X_train_acc = np.array(quantum_train_acc)
Y_train_acc = np.array(non_equiv_train_acc)
print(f"Mean baseline train acc   : {X_train_acc.mean():.3f} ± {X_train_acc.std():.3f}")
print(f"Mean new ansatz train acc : {Y_train_acc.mean():.3f} ± {Y_train_acc.std():.3f}")
print()

X_test_acc = np.array(quantum_test_acc)
Y_test_acc = np.array(non_equiv_test_acc)
print(f"Mean baseline test acc   : {X_test_acc.mean():.3f} ± {X_test_acc.std():.3f}")
print(f"Mean new ansatz test acc : {Y_test_acc.mean():.3f} ± {Y_test_acc.std():.3f}")
print()

X_novel_test_acc = np.array(quantum_novel_test_acc)
Y_novel_test_acc = np.array(non_equiv_novel_test_acc)
print(f"Mean baseline novel test acc   : {X_novel_test_acc.mean():.3f} ± {X_novel_test_acc.std():.3f}")
print(f"Mean new ansatz novel test acc : {Y_novel_test_acc.mean():.3f} ± {Y_novel_test_acc.std():.3f}")
print()

X_total_time = np.array(quantum_total_time)
Y_total_time = np.array(non_equiv_total_time)
print(f"Mean baseline total_time   : {X_total_time.mean():.3f} ± {X_total_time.std():.3f}")
print(f"Mean new ansatz total_time : {Y_total_time.mean():.3f} ± {Y_total_time.std():.3f}")

X_avg_time = np.array(quantum_avg_time)
Y_avg_time = np.array(non_equiv_avg_time)
print(f"Mean baseline avg_time   : {X_avg_time.mean():.3f} ± {X_avg_time.std():.3f}")
print(f"Mean new ansatz avg_time : {Y_avg_time.mean():.3f} ± {Y_avg_time.std():.3f}")
