import utils, kernels

dataset_train = utils.KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
dataset_test = utils.KernelDataset(x_csv_path="./Xte.csv")

dataloader_train = utils.KernelDataLoader(dataset_test)
