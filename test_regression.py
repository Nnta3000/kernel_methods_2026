from typing import Optional

from jax import Array
from jaxtyping import Integer
import kernels, classifiers
import jax.numpy as jnp
from utils.classification import OneVsAllStrategy, Trainer
from utils.kernelisation import KernelDataset, KernelDataLoader
from utils.tools import prediction_to_csv

if __name__ == "__main__":
    dataset_train = KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
    dataset_test  = KernelDataset(x_csv_path="./Xte.csv")

    # Gretton et al. (2012) — "A Kernel Two-Sample Test" to find the best sigma
    optimize_sigma = False 
    sigma = 2000 
    if optimize_sigma:
        X_sub = dataset_train.images[:500].reshape(-1, 3072)
        dists = jnp.array([[jnp.sum((X_sub[i] - X_sub[j])**2)
                            for j in range(len(X_sub))]
                            for i in range(len(X_sub))])
        sigma = float(jnp.sqrt(jnp.median(dists) / 2))
        print(f"Median heuristic sigma: {sigma:.2f}")

    kernels = {"Linear": kernels.LinearKernel(),
               "Gaussian": kernels.PolynomialKernel(degree=3, c=1.0) ,
               "Laplacian": kernels.LaplacianKernel(sigma=sigma),
               "Histogram": kernels.HistogramIntersectionKernel(),
               "Anova": kernels.AnovaKernel(sigma=sigma, p=2),
               "ArcCosine": kernels.ArcCosineKernel() ,
            }
     
    kernel_name =  "Anova"
    dataloader_full = KernelDataLoader(dataset_train, kernels[kernel_name], max_size=5000)

    clf = Trainer( dataloader_full)
    clf.fit()

    predictions : Optional[Integer[Array, "M"]]= None    
    best_score = 0.38 # Anova kernel with KRR, p=2, sigma=2000, lambda=1000
    lambdas = [4000.0, 2000.0,1000.0]
    for lam in lambdas:
        reg = classifiers.Regularizer(lam=lam)
        print(f"\n--- λ={lam} ---")

        for name, factory in [
            ("SVM", lambda reg=reg: classifiers.KernelSVM(reg, n_iter=100)),
            ("KRR", lambda reg=reg: classifiers.KernelRidgeRegression(reg)),
            ("KLR", lambda reg=reg: classifiers.KernelLogisticRegression(reg, n_iter=20)),
        ]:
            clf.strategy = OneVsAllStrategy(n_classes=10, model_factory=factory)
            clf.refit()  # only redoes the linear solve, not the kernel
            print(f"  {name}", end=" ")
            score = clf.score()[1]
            if score >= best_score:
                score = best_score
                prediction = clf.predict(dataset_test)
    
    assert prediction is not None, "No better prediction was found"
    print(f"The best contender scored {best_score}")
    prediction_to_csv(prediction) 
