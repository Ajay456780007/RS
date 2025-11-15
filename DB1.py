import numpy as np
import os


def generate_metrics_dataset(db):
    base_dir = "Analysis"
    os.makedirs(base_dir, exist_ok=True)

    models = ['KNN', 'CNN_Resnet', 'CNN', 'SVM', 'HGNN', 'PM_WA', 'DiT', 'PM']
    metrics = ['ACC', 'SEN', 'SPE', 'F1score', 'REC', 'PRE', 'TPR', 'FPR']
    epochs_list = [100, 200, 300, 400, 500]
    training_percentages = [40, 50, 60, 70, 80, 90]
    metric_ranges = {
        'ACC': (0.8867, 0.9789),
        'SEN': (0.8789, 0.9500),
        'SPE': (0.8686, 0.9898),
        'F1score': (0.87676, 0.9678),
        'REC': (0.8577, 0.9746),
        'PRE': (0.9055, 0.9606),
        'TPR': (0.4067, 0.9647),
        'FPR': (0.00, 0.0208)
    }

    def enforce_epoch_progression(prev_data, low, high):
        steps = 6
        base = np.zeros(steps)

        if prev_data is None:
            x = np.linspace(1, steps, steps)
            progression = low + (high - low) * ((x / steps) ** 1.6)
            noise = np.random.uniform(-0.01, 0.01, steps)
            base = np.clip(np.round(progression + noise, 4), low, high)
        else:
            for i in range(steps):
                # Calculate how close we are to the high bound
                closeness = (prev_data[i] - low) / (high - low + 1e-8)
                max_step = 0.08 if closeness > 0.8 else 0.02
                min_step = 0.04

                step = np.random.uniform(min_step, max_step)
                val = prev_data[i] + step

                # Ensure we don't exceed upper bound
                if i == steps - 1:
                    val = min(val, high - np.random.uniform(0.003, 0.060))
                else:
                    val = min(val, high - np.random.uniform(0.001, 0.03))

                base[i] = np.clip(round(val, 4), low, high)

        return np.round(base, 4)

    def enforce_proposed_model_highest(metric, metric_range):
        for j in range(metric.shape[1]):
            max_val = np.max(metric[:-1, j])
            if metric[-1, j] <= max_val:
                metric[-1, j] = round(min(max_val + np.random.uniform(0.005, 0.015), metric_range[1]), 4)
        return metric

    def recalculate_accuracy_f1(sen, spe, pre, rec):
        acc = np.round((sen + spe) / 2, 4)
        f1 = np.round(2 * (pre * rec) / (pre + rec + 1e-8), 4)
        return acc, f1

    def validate_metrics(acc, sen, spe, f1, pre, rec):
        for j in range(acc.shape[1]):
            acc[:, j] = np.round((sen[:, j] + spe[:, j]) / 2, 4)
            rec[:, j] = sen[:, j]
            f1[:, j] = np.round(2 * (pre[:, j] * rec[:, j]) / (pre[:, j] + rec[:, j] + 1e-8), 4)
        return acc, rec, f1

    def save_metrics(path, metrics_dict):
        for key, value in metrics_dict.items():
            np.save(os.path.join(path, f"{key}.npy"), value)

    # === PERFORMANCE ANALYSIS ===
    perf_dir = f"{base_dir}/Performance_Analysis/Concated_epochs/{db}/"
    os.makedirs(perf_dir, exist_ok=True)
    perf_values_by_epoch = {}
    prev_epoch_data = None

    for epoch in epochs_list:
        data = np.zeros((len(metrics), len(training_percentages)))
        if epoch != 500:
            for i, metric in enumerate(metrics):
                low, high = metric_ranges[metric]
                values = enforce_epoch_progression(prev_epoch_data[i] if prev_epoch_data is not None else None, low,
                                                   high)
                data[i] = values
            prev_epoch_data = data
        perf_values_by_epoch[epoch] = data
    for epoch in epochs_list:
        print(f"Epoch {epoch} - 90% values: {[val[-1] for val in perf_values_by_epoch[epoch]]}")
    # === COMPARATIVE ANALYSIS ===
    comp_dir = f"{base_dir}/Comparative_Analysis/{db}/"
    os.makedirs(comp_dir, exist_ok=True)
    rows, cols = len(models), len(training_percentages)
    # model_registry = {
    #     "BeiT": BeiT,
    #     "CNN": CNN,
    #     "Darknet_53_CNN": darknet_53_cnn,
    #     "DC_GAN_MDFC_Resnet": DC_GAN_MDFC_ResNet,
    #     "dCNN": dCNN,
    #     "S2AFS": S2AFS,
    #     "SVM": SVM
    # }

    model_metric_ranges = {
        "ACC": [(0.91, 0.94), (0.89, 0.95), (0.90, 0.95), (0.90, 0.93), (0.90, 0.95), (0.90, 0.95), (0.91, 0.96),
                (0.91, 0.9794)],
        "SEN": [(0.89, 0.93), (0.88, 0.93), (0.89, 0.94), (0.88, 0.94), (0.87, 0.92), (0.91, 0.95), (0.90, 0.95),
                (0.91, 0.9699)],
        "SPE": [(0.90, 0.95), (0.87, 0.978), (0.88, 0.97), (0.87, 0.95), (0.89, 0.96), (0.89, 0.97), (0.90, 0.95),
                (0.91, 0.9889)],
        "PRE": [(0.90, 0.95), (0.88, 0.95), (0.91, 0.94), (0.87, 0.96), (0.87, 0.96), (0.89, 0.97), (0.91, 0.95),
                (0.91, 0.9709)],
        "TPR": [(0.41, 0.90), (0.50, 0.93), (0.57, 0.92), (0.56, 0.92), (0.67, 0.90), (0.83, 0.96), (0.89, 0.95),
                (0.91, 0.9706)],
        "FPR": [(0.00, 0.02), (0.00, 0.14), (0.00, 0.0797), (0.00, 0.17), (0.00, 0.089), (0.00, 0.07), (0.00, 0.05),
                (0.00, 0.023)]
    }

    model_ranges = [
        (0.9045, 0.9787),  # Base_model GPT2
        (0.8332, 0.9567),  # Bert
        (0.8478, 0.9378),  # CNN_BiLSTM
        (0.8524, 0.9143),  # BiLSTM
        (0.8132, 0.9454),  # DGCNN
        (0.8676, 0.9078),  # FGSACML
        (0.8557, 0.9477),  # PM_WCL
        (0.8478, 0.9841)  # Proposed_model
    ]

    def generate_model_metrics(metric_name, cols=6):
        ranges = model_metric_ranges[metric_name]
        data = np.zeros((rows, cols))
        for i, (low, high) in enumerate(ranges):
            data[i] = np.sort(np.round(np.random.uniform(low, high, cols), 4))
        return data

    SEN = generate_model_metrics("SEN")
    SPE = generate_model_metrics("SPE")
    PRE = generate_model_metrics("PRE")
    TPR = generate_model_metrics("TPR")
    FPR = generate_model_metrics("FPR")

    # SEN = generate_model_metrics(model_ranges)
    REC = SEN.copy()
    # SPE = generate_model_metrics(model_ranges)
    # PRE = generate_model_metrics(model_ranges)
    # TPR = generate_model_metrics([(0.4, 0.9)] * rows)
    # FPR = generate_model_metrics([(0.0, 0.3)] * rows)
    ACC, F1 = recalculate_accuracy_f1(SEN, SPE, PRE, REC)

    for metric_array, key in zip([SEN, SPE, PRE, TPR, FPR], ['SEN', 'SPE', 'PRE', 'TPR', 'FPR']):
        enforce_proposed_model_highest(metric_array, (0.0, 1.0))

    REC = SEN.copy()
    ACC, F1 = recalculate_accuracy_f1(SEN, SPE, PRE, REC)

    comp_metrics = {
        "ACC_1": ACC, "SEN_1": SEN, "REC_1": REC, "SPE_1": SPE,
        "PRE_1": PRE, "F1score_1": F1, "TPR_1": TPR, "FPR_1": FPR,
    }
    save_metrics(comp_dir, comp_metrics)

    # 500 EPOCHS COPY FROM COMPARATIVE LAST ROW
    # 500 EPOCHS COPY FROM COMPARATIVE LAST ROW
    epoch500 = np.zeros((len(metrics), len(training_percentages)))  # <-- fix
    for i, metric in enumerate(metrics):
        epoch500[i] = comp_metrics[f"{metric}_1"][-1]
    perf_values_by_epoch[500] = epoch500
    np.save(os.path.join(perf_dir, f"metrics_epochs_500.npy"), epoch500)

    # === KF ANALYSIS ===
    def generate_kf_analysis():
        kf_dir = f"{base_dir}/KF_Analysis/{db}/"
        os.makedirs(kf_dir, exist_ok=True)
        kf_cols = 5

        def gen_range_data(ranges):
            data = np.zeros((rows, kf_cols))
            for i, (low, high) in enumerate(ranges):
                data[i] = np.sort(np.round(np.random.uniform(low, high, kf_cols), 4))
            return data

        SEN = gen_range_data(model_ranges)
        REC = SEN.copy()
        SPE = gen_range_data(model_ranges)
        PRE = gen_range_data(model_ranges)
        ACC, F1 = recalculate_accuracy_f1(SEN, SPE, PRE, REC)

        for metric_array in [SEN, SPE, PRE]:
            enforce_proposed_model_highest(metric_array, (0.0, 1.0))

        ACC, F1 = recalculate_accuracy_f1(SEN, SPE, PRE, REC)
        kf_metrics = {
            "ACC_2": ACC, "SEN_2": SEN, "REC_2": REC, "SPE_2": SPE,
            "PRE_2": PRE, "F1score_2": F1,
        }
        save_metrics(kf_dir, kf_metrics)
        return kf_metrics

    kf_metrics = generate_kf_analysis()

    # === FINAL VALIDATION ===
    def validate_all():
        acc, rec, f1 = validate_metrics(comp_metrics["ACC_1"], comp_metrics["SEN_1"],
                                        comp_metrics["SPE_1"], comp_metrics["F1score_1"],
                                        comp_metrics["PRE_1"], comp_metrics["REC_1"])
        comp_metrics["ACC_1"], comp_metrics["REC_1"], comp_metrics["F1score_1"] = acc, rec, f1

        acc, rec, f1 = validate_metrics(kf_metrics["ACC_2"], kf_metrics["SEN_2"],
                                        kf_metrics["SPE_2"], kf_metrics["F1score_2"],
                                        kf_metrics["PRE_2"], kf_metrics["REC_2"])
        kf_metrics["ACC_2"], kf_metrics["REC_2"], kf_metrics["F1score_2"] = acc, rec, f1

        for epoch in epochs_list:
            mat = perf_values_by_epoch[epoch]
            acc, rec, f1 = validate_metrics(mat[0:1], mat[1:2], mat[2:3], mat[3:4], mat[5:6], mat[4:5])
            mat[0], mat[4], mat[3] = acc[0], rec[0], f1[0]
            perf_values_by_epoch[epoch] = mat
            np.save(os.path.join(perf_dir, f"metrics_epochs_{epoch}.npy"), mat)

        save_metrics(f"{base_dir}/Comparative_Analysis/{db}/", comp_metrics)
        save_metrics(f"{base_dir}/KF_Analysis/{db}/", kf_metrics)

    # === FINAL STRICT ENFORCEMENT ===
    def enforce_strict_epoch_progression():
        reference = perf_values_by_epoch[500].copy()  # Shape: (8 metrics, 6 training %)
        retries = 0
        max_retries = 3

        while retries < max_retries:
            retry_needed = False
            for epoch in [100, 200, 300, 400]:
                data = perf_values_by_epoch[epoch]
                new_data = data.copy()

                for i in range(len(metrics)):  # For each metric row
                    for j in range(len(training_percentages)):  # For each column
                        if new_data[i][j] >= reference[i][j]:
                            # Reduce value below reference
                            new_val = reference[i][j] - np.random.uniform(0.003, 0.02)
                            new_data[i][j] = round(max(new_val, 0), 4)
                            retry_needed = True

                perf_values_by_epoch[epoch] = new_data
                np.save(os.path.join(perf_dir, f"metrics_epochs_{epoch}.npy"), new_data)

            validate_all()

            if retry_needed:
                print(f"🔁 Retry strict enforcement attempt {retries + 1}")
                retries += 1
            else:
                print("✅ Strict progression validated successfully against epoch 500.")
                return

        print("⚠️ Maximum retries reached. Metrics adjusted, but some values may still need review.")

    enforce_strict_epoch_progression()

    validate_all()
    print("✅ All analyses generated, validated, and saved successfully.")


# # Run the function
generate_metrics_dataset("AFDB")
a = np.load("Temp/Analysis/Comparative_Analysis/AFDB/ACC_1.npy")
print("The last row is:", a[-1])

# hERE fIRST UNSERSTAND THE CODE WELL , her comparative is calculated and then performance and then kf anslysis , the performance anlysis alst row and the comparative
# Analysis last row is made equal , now like wise i also need to add the kf Analyss performance values read the values from the files where the
# kf anlysis value is saves  files like ACC_2.npy, SEN_2.npy, SPE_2.npy, PRE_2.npy, F1score_2.npy, REC_2.npy, and then read the last row and make the last row same for both the kf fold comparative  currently and for kf fold performance
# Then the remaining row valus in the performance kf fold should be random in the range given slowly increasing from 6 to 10 folds , give me the complete code with kf fold perforance
# values and save inside the Analysis/KF_Perf/{DB}/metrics_100_epochs , 200 epochs ,300 epochs,400_epochs,500_epochs , then each row is each metrics ,generate values for performance kf like the normal compartive and performance values generated the values should be
# there we generate 6 columns here 5 columns , h=give me the complete corrected code
# column is the folds from 6 to 10 and each row is the
# generate_metrics_dataset("Malayalam")
# generate_metrics_dataset("Kannada")

# even though i do so in performance values why the 500 epochs values are lesser in some area my condition is always 100 epochs file values less than 200 epochs file values and 300 epochs file values less than 400 and  400 less than 500 so always my 500 epochs file values should be high right why the values lower check

# why in performance Analysis values in all epochs files why the 90 percentage values for files looks same
