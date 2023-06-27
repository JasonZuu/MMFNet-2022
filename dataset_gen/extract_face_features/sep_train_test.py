import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./IF_Features.csv", index_col=None, sep=",")
    train_data = data[:120000]
    test_data = data[120000:]
    train_data.to_csv("./IF_train.csv", sep=",", index=None)
    test_data.to_csv("./IF_test.csv", sep=",", index=None)
