import kernels, classifiers
import jax.numpy as jnp
from utils.classification import MulticlassStrategy, OneVsAllStrategy, Classifier
from utils.kernelisation import KernelDataset, KernelDataLoader

if __name__ == "__main__":
    dataset_train = KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
    dataset_test  = KernelDataset(x_csv_path="./Xte.csv")

    # Gretton et al. (2012) — "A Kernel Two-Sample Test" to find the best sigma
    optimize_sigma = False 
    sigma = 55.40
    if optimize_sigma:
        X_sub = dataset_train.images[:500].reshape(-1, 3072)
        dists = jnp.array([[jnp.sum((X_sub[i] - X_sub[j])**2)
                            for j in range(len(X_sub))]
                            for i in range(len(X_sub))])
        sigma = float(jnp.sqrt(jnp.median(dists) / 2))
        print(f"Median heuristic sigma: {sigma:.2f}")

    # kernel =  kernels.PolynomialKernel(degree=4, c=1.0) 
    # kernel = kernels.LaplacianKernel(sigma=sigma)
    # kernel = kernels.GaussianKernel(sigma=sigma)
    kernel = kernels.HistogramIntersectionKernel()
    # kernel = kernels.ArcCosineKernel()
    dataloader_full = KernelDataLoader(dataset_train, kernel, max_size=5000)

    clf = Classifier( dataloader_full)
    clf.fit()  # computes K_train, K_val once

    lambdas = [0, 0.01, 0.1, 1.0, 10.0]
    for lam in lambdas:
        reg = classifiers.Regularizer(lam=lam)
        print(f"\n--- λ={lam} ---")

        for name, factory in [
            ("KRR", lambda reg=reg: classifiers.KernelRidgeRegression(reg)),
            ("KLR", lambda reg=reg: classifiers.KernelLogisticRegression(reg, n_iter=20)),
        ]:
            clf.strategy = OneVsAllStrategy(n_classes=10, model_factory=factory)
            clf.refit()  # only redoes the linear solve, not the kernel
            print(f"  {name}", end=" ")
            clf.score()
