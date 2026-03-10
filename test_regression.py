import utils, kernels, regression
import jax.numpy as jnp

if __name__ == "__main__":
    dataset_train = utils.KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
    dataset_test  = utils.KernelDataset(x_csv_path="./Xte.csv")

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
    # kernel = kernels.HistogramIntersectionKernel()
    kernel = kernels.ArcCosineKernel()
    dataloader_full = utils.KernelDataLoader(dataset_train, kernel, max_size=5000)


    # Compute full kernel matrix once, then split by submatrix indexing
    dataloader_train, dataloader_val = dataloader_full.split(test_size=0.2)
    K_train = dataloader_train.get_kernel_matrix()
    K_val   = dataloader_val.get_kernel_matrix(Y=dataloader_train.dataset.images.reshape(-1, 3072))
    y_train = dataloader_train.dataset.labels
    y_val   = dataloader_val.dataset.labels
    
    print(f"K_train — min: {K_train.min():.4f}, max: {K_train.max():.4f}, mean: {K_train.mean():.4f}")
    assert y_train is not None
    print(f"y_train unique: {jnp.unique(y_train)}, dtype: {y_train.dtype}")
    
    lambdas = [0, 0.01, 0.1, 1.0, 10.0]
    for lam in lambdas:
        reg = regression.Regularizer(lam=lam)
        print(f"\n--- λ={lam} ---")

        # KRR
        krr_strategy = regression.OneVsAllStrategy(
            n_classes=10,
            model_factory=lambda reg=reg: regression.KernelRidgeRegression(reg)
        )
        krr_clf = regression.Classifier(krr_strategy)
        krr_clf.fit(K_train, y_train)
        krr_train = jnp.mean(krr_clf.predict(K_train) == y_train)
        krr_val   = jnp.mean(krr_clf.predict(K_val)   == y_val)
        print(f"  KRR  — train: {krr_train:.4f}, val: {krr_val:.4f}")

        # KLR
        klr_strategy = regression.OneVsAllStrategy(
            n_classes=10,
            model_factory=lambda reg=reg: regression.KernelLogisticRegression(reg, n_iter=20)
        )
        klr_clf = regression.Classifier(klr_strategy)
        klr_clf.fit(K_train, y_train)
        klr_train = jnp.mean(klr_clf.predict(K_train) == y_train)
        klr_val   = jnp.mean(klr_clf.predict(K_val)   == y_val)
        print(f"  KLR  — train: {klr_train:.4f}, val: {klr_val:.4f}")
