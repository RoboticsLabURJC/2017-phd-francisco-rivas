import os



def load_best_model(input_path: str):
    current_ws = os.path.join(input_path, "checkpoints")

    candidates = []
    for candidate in os.listdir(current_ws):
        if "train" in candidate or "best" in candidate:
            continue
        loss = float(".".join(candidate.split("=")[-1].split(".")[0:-1]).replace("-v1", ""))
        candidates.append((loss, os.path.join(current_ws, candidate)))

    candidates_sorted = sorted(candidates, key=lambda tup: tup[0])

    return candidates_sorted[0][1]


if __name__ == "__main__":
    test_input = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_14"

    a = load_best_model(test_input)
    print(a)