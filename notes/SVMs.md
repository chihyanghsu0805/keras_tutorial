#   Support Vector Machines (SVMs):

SVMs are `optimal margin classifiers` with `kernel tricks`.

-   Optimal Margin Classifiers: For binary classification problems, the labels y become [-1, 1] and θ is represented by w and b. The hypothesis becomes h<sub>w,b</sub>(x) = g(w<sup>T</sup>x+b).

    -   The `Functional Margin` (FM) is defined as Γ<sup>i</sup><sub>FM</sub> = y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b). If y = 1, w<sup>T</sup>x<sup>i</sup>+b >> 0 and if y = -1, w<sup>T</sup>x<sup>i</sup>+b << 0. For the entire train set, Γ<sub>FM</sub> = min Γ<sup>i</sup><sub>FM</sub>. Note that FM is not a good metric of confidence since its magnitude is directly affected by w and b.

    -   The `Geometric Margin` (GM) is defined as Γ<sup>i</sup><sub>GM</sub> = y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) / ||w|| and Γ<sub>GM</sub> = min Γ<sup>i</sup><sub>GM</sub>.

    -   The optimal margin classifier seeks to max<sub>Γ, w, b</sub> Γ<sub>GM</sub> subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) / ||w|| >= Γ<sub>GM</sub> and ||w|| = 1. It can be written as min<sub>w, b</sub> (1/2) *||w||<sup>2</sup> subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) >= 1 when FM is scaled to be 1 or ||w|| = 1 / Γ<sub>GM</sub>. This optimization can be solved with `Quadratic Programming`.

-   Kernel Tricks: The parameters w can been seen as adding some multiples of x, so the optimization problem can be written as min<sub>w, b</sub> (1/2) * (Σ<sub>i</sub>a<sub>i</sub>x<sup>i</sup>y<sup>i</sup>)<sup>T</sup>(Σ<sub>j</sub>a<sub>j</sub>x<sup>j</sup>y<sup>j</sup>) subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) >= 1. Expanding the matrix multiplication, min<sub>w, b</sub> (1/2) * (Σ<sub>i</sub>Σ<sub>j</sub>a<sub>i</sub>a<sub>j</sub>y<sup>i</sup>y<sup>j</sup>x<sup>i</sup><sup>T</sup>x<sup>j</sup>) where x<sup>i</sup><sup>T</sup>x<sup>j</sup> is the inner product <x<sup>i</sup>, x<sup>j</sup>>, or kernel K(x, z). Common kernels are,

    -   `Linear Kernel` K(x, z) = x<sup>T</sup>z
    -   `Gaussian Kernel` K(x, z) = exp(-||x-z||<sup>2</sup> / (2σ<sup>2</sup>))
    -   `Polynomial Kernel` K(x, z) = (x<sup>T</sup>z)<sup>d</sup>

-   Regularization: For non-linear separable cases, `L1-norm soft margin` can be used with the C parameter controlling the functional margin error.
