# Gradient descent

Source: Wikipedia, article 'Gradient descent' (public, CC BY-SA)

Gradient descent is a method for unconstrained mathematical optimization. It is a first-order iterative algorithm for minimizing a differentiable multivariate function.

The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a trajectory that maximizes that function; the procedure is then known as gradient ascent.
Gradient descent should not be confused with local search algorithms, although both are iterative methods for optimization.
Gradient descent is particularly useful in machine learning and artificial intelligence for minimizing the cost or loss function. 
Gradient descent is generally attributed to Augustin-Louis Cauchy, who first suggested it in 1847. Jacques Hadamard independently proposed a similar method in 1907. Its convergence properties for non-linear optimization problems were first studied by Haskell Curry in 1944, with the method becoming increasingly well-studied and used in the following decades.
A simple extension of gradient descent, stochastic gradient descent, serves as the most basic algorithm used for training most deep networks today.


== Description ==

Gradient descent is based on the observation that if the multi-variable function f(𝐱) is defined and differentiable in a neighborhood of a point 𝐚, then f(𝐱) decreases fastest if one goes from 𝐚 in the direction of the negative gradient of f at 𝐚,i.e.,−∇f(𝐚). It follows that, if

𝐚n+1=𝐚n−η∇f(𝐚n)
for a small enough step size or learning rate η∈ℝ+, then  f(𝐚𝐧)≥f(𝐚𝐧+𝟏). In other words, the term η∇f(𝐚) is subtracted from 𝐚 because we want to move against the gradient, toward the local minimum. With this observation in mind, one starts with a guess 𝐱0 for a local minimum of f, and considers the sequence 𝐱0,𝐱1,𝐱2,… such that

𝐱n+1=𝐱n−ηn∇f(𝐱n), n≥0.
We have a monotonic sequence

f(𝐱0)≥f(𝐱1)≥f(𝐱2)≥⋯,
so the sequence (𝐱n) converges to the desired local minimum. Note that the value of the step size η is allowed to change at every iteration. 
It is possible to guarantee the convergence to a local minimum under certain assumptions on the function f (for example, f convex and ∇f Lipschitz) and particular choices of η. Those include the sequence
ηn=|(𝐱n−𝐱n−1)⊤[∇f(𝐱n)−∇f(𝐱n−1)]|‖∇f(𝐱n)−∇f(𝐱n−1)‖2
as in the Barzilai-Borwein method, or a sequence ηn satisfying the Wolfe conditions (which can be found by using line search). When the function f is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.
This process is illustrated in the adjacent picture. Here, f is assumed to be defined on the plane, and that its graph has a bowl shape.  The blue curves are the contour lines, that is, the regions on which the value of f is constant. A red arrow originating at a point shows the direction of the negative gradient at that point. Note that the (negative) gradient at a point is orthogonal to the contour line going through that point. We see that gradient descent leads us to the bottom of the bowl, that is, to the point where the value of the function f is minimal.


=== An analogy for understanding gradient descent ===

The basic intuition behind gradient descent can be illustrated by a hypothetical scenario. People are stuck in the mountains and are trying to get down (i.e., trying to find the global minimum). There is heavy fog such that visibility is extremely low. Therefore, the path down the mountain is not visible, so they must use local information to find the minimum. They can use the method of gradient descent, which involves looking at the steepness of the hill at their current position, then proceeding in the direction with the steepest descent (i.e., downhill). If they were trying to find the top of the mountain (i.e., the maximum), then they would proceed in the direction of steepest ascent (i.e., uphill). Using this method, they would eventually find their way down the mountain or possibly get stuck in some hole (i.e., local minimum or saddle point), like a mountain lake. However, assume also that the steepness of the hill is not immediately obvious with simple observation, but rather it requires a sophisticated instrument to measure, which the people happen to have at that moment. It takes quite some time to measure the steepness of the hill with the instrument. Thus, they should minimize their use of the instrument if they want to get down the mountain before sunset. The difficulty then is choosing the frequency at which they should measure the steepness of the hill so as not to go off track.
In this analogy, the people represent the algorithm, and the path taken down the mountain represents the sequence of parameter settings that the algorithm will explore. The steepness of the hill represents the slope of the function at that point. The instrument used to measure steepness is differentiation. The direction they choose to travel in aligns with the gradient of the function at that point. The amount of time they travel before taking another measurement is the step size.


=== Choosing the step size and descent direction ===
Since using a step size η that is too small would slow convergence, and a η too large would lead to overshoot and divergence, finding a good setting of η is an important practical problem. Philip Wolfe also advocated using "clever choices of the [descent] direction" in practice. While using a direction that deviates from the steepest descent direction may seem counter-intuitive, the idea is that the smaller slope may be compensated for by being sustained over a much longer distance.
To reason about this mathematically, consider a direction 𝐩n and step size ηn and consider the more general update:

𝐚n+1=𝐚n−ηn𝐩n.
Finding good settings of 𝐩n and ηn requires some thought. First of all, we would like the update direction to point downhill. Mathematically, letting θn denote the angle between −∇f(𝐚𝐧) and 𝐩n, this requires that cos⁡θn>0. To say more, we need more information about the objective function that we are optimising. Under the fairly weak assumption that f is continuously differentiable, we may prove that:

This inequality implies that the amount by which we can be sure the function f is decreased depends on a trade off between the two terms in square brackets. The first term in square brackets measures the angle between the descent direction and the negative gradient. The second term measures how quickly the gradient changes along the descent direction.
In principle inequality (1) could be optimized over 𝐩n and ηn to choose an optimal step size and direction. The problem is that evaluating the second term in square brackets requires evaluating ∇f(𝐚n−tηn𝐩n), and extra gradient evaluations are generally expensive and undesirable. Some ways around this problem are:

Forgo the benefits of a clever descent direction by setting 𝐩n=∇f(𝐚𝐧), and use line search to find a suitable step-size γn, such as one that satisfies the Wolfe conditions. A more economic way of choosing learning rates is backtracking line search, a method that has both good theoretical guarantees and experimental results. Note that one does not need to choose  𝐩n to be the gradient; any direction that has positive inner product with the gradient will result in a reduction of the function value (for a sufficiently small value of ηn).
Assuming that f is twice-differentiable, use its Hessian ∇2f to estimate ‖∇f(𝐚n−tηn𝐩n)−∇f(𝐚n)‖2≈‖tηn∇2f(𝐚n)𝐩n‖.Then choose 𝐩n and ηn by optimising inequality (1).
Assuming that ∇f is Lipschitz, use its Lipschitz constant L to bound ‖∇f(𝐚n−tηn𝐩n)−∇f(𝐚n)‖2≤Ltηn‖𝐩n‖. Then choose 𝐩n and ηn by optimising inequality (1).
Build a custom model of maxt∈[0,1]‖∇f(𝐚n−tηn𝐩n)−∇f(𝐚n)‖2‖∇f(𝐚n)‖2 for f. Then choose 𝐩n and ηn by optimising inequality (1).
Under stronger assumptions on the function f such as convexity, more advanced techniques may be possible.
Usually by following one of the recipes above, convergence to a local minimum can be guaranteed. When the function f is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.


== Solution of a linear system ==

Gradient descent can be used to solve a system of linear equations

𝐀𝐱−𝐛=0
reformulated as a quadratic minimization problem.
If the system matrix 𝐀 is real symmetric and positive-definite, an objective function is defined as the quadratic function, with minimization of

f(𝐱)=𝐱⊤𝐀𝐱−2𝐱⊤𝐛,
so that

∇f(𝐱)=2(𝐀𝐱−𝐛).
For a general real matrix 𝐀, linear least squares define

f(𝐱)=‖𝐀𝐱−𝐛‖2.
In traditional linear least squares for real 𝐀 and 𝐛 the Euclidean norm is used, in which case

∇f(𝐱)=2𝐀⊤(𝐀𝐱−𝐛).
The line search minimization, finding the locally optimal step size η on every iteration, can be performed analytically for quadratic functions, and explicit formulas for the locally optimal η are known.
For example, for real symmetric and positive-definite matrix 𝐀, a simple algorithm can be as follows,

repeat in the loop:𝐫:=𝐛−𝐀𝐱η:=𝐫⊤𝐫/𝐫⊤𝐀𝐫𝐱:=𝐱+η𝐫if 𝐫⊤𝐫 is sufficiently small, then exit loopend repeat loopreturn 𝐱 as the result
To avoid multiplying by 𝐀 twice per iteration,
we note that 𝐱:=𝐱+η𝐫 implies 𝐫:=𝐫−η𝐀𝐫, which gives the traditional algorithm,

𝐫:=𝐛−𝐀𝐱repeat in the loop:η:=𝐫⊤𝐫/𝐫⊤𝐀𝐫𝐱:=𝐱+η𝐫if 𝐫⊤𝐫 is sufficiently small, then exit loop𝐫:=𝐫−η𝐀𝐫end repeat loopreturn 𝐱 as the result
The method is rarely used for solving linear equations, with the conjugate gradient method being one of the most popular alternatives. The number of gradient descent iterations is commonly proportional to the spectral condition number κ(𝐀) of the system matrix 𝐀 (the ratio of the maximum to minimum eigenvalues of 𝐀⊤𝐀), while the convergence of conjugate gradient method is typically determined by a square root of the condition number, i.e., is much faster. Both methods can benefit from preconditioning, where gradient descent may require less assumptions on the preconditioner.


=== Geometric behavior and residual orthogonality ===
In steepest descent applied to solving 𝐀𝐱=𝐛, where 𝐀 is symmetric positive-definite, the residual vectors 𝐫k=𝐛−𝐀𝐱k are orthogonal across iterations:

⟨𝐫k+1,𝐫k⟩=0.
Because each step is taken in the steepest direction, steepest-descent steps alternate between directions aligned with the extreme axes of the elongated level sets.  When κ(𝐀) is large, this produces a characteristic zig–zag path. The poor conditioning of 𝐀 is the primary cause of the slow convergence, and orthogonality of successive residuals reinforces this alternation.
As shown in the image on the right, steepest descent converges slowly due to the high condition number of 𝐀, and the orthogonality of residuals forces each new direction to undo the overshoot from the previous step. The result is a path that zigzags toward the solution. This inefficiency is one reason conjugate gradient or preconditioning methods are preferred.


== Solution of a non-linear system ==
Gradient descent can also be used to solve a system of nonlinear equations. Below is an example that shows how to use the gradient descent to solve for three unknown variables, x1, x2, and x3. This example shows one iteration of the gradient descent.
Consider the nonlinear system of equations

{3x1−cos⁡(x2x3)−32=04x12−625x22+2x2−1=0exp⁡(−x1x2)+20x3+10π−33=0
Let us introduce the associated function

G(𝐱)=[3x1−cos⁡(x2x3)−324x12−625x22+2x2−1exp⁡(−x1x2)+20x3+10π−33],
where

𝐱=[x1x2x3].
One might now define the objective function

f(𝐱)=12G⊤(𝐱)G(𝐱)=12[(3x1−cos⁡(x2x3)−32)2+(4x12−625x22+2x2−1)2+(exp⁡(−x1x2)+20x3+10π−33)2],
which we will attempt to minimize. As an initial guess, let us use

𝐱(0)=𝟎=[000].
We know that

𝐱(1)=𝟎−η0∇f(𝟎)=𝟎−η0JG(𝟎)⊤G(𝟎),
where the Jacobian matrix JG is given by

JG(𝐱)=[3sin⁡(x2x3)x3sin⁡(x2x3)x28x1−1250x2+20−x2exp⁡(−x1x2)−x1exp⁡(−x1x2)20].
We calculate:

JG(𝟎)=[3000200020],G(𝟎)=[−2.5−110.472].
Thus

𝐱(1)=𝟎−η0[−7.5−2209.44],
and

f(𝟎)=0.5((−2.5)2+(−1)2+(10.472)2)=58.456.
Now, a suitable η0 must be found such that

f(𝐱(1))≤f(𝐱(0))=f(𝟎).
This can be done with any of a variety of line search algorithms. One might also simply guess η0=0.001, which gives

𝐱(1)=[0.00750.002−0.20944].
Evaluating the objective function at this value, yields

f(𝐱(1))=0.5((−2.48)2+(−1.00)2+(6.28)2)=23.306.
The decrease from f(𝟎)=58.456 to the next step's value of

f(𝐱(1))=23.306
is a sizable decrease in the objective function. Further steps would reduce its value further until an approximate solution to the system was found.


== Comments ==
Gradient descent works in spaces of any number of dimensions, even in infinite-dimensional ones. In the latter case, the search space is typically a function space, and one calculates the Fréchet derivative of the functional to be minimized to determine the descent direction.
That gradient descent works in any number of dimensions (finite number at least) can be seen as a consequence of the Cauchy–Schwarz inequality, i.e. the magnitude of the inner (dot) product of two vectors of any dimension is maximized when they are colinear. In the case of gradient descent, that would be when the vector of independent variable adjustments is proportional to the gradient vector of partial derivatives.
The gradient descent can take many iterations to compute a local minimum with a required accuracy, if the curvature in different directions is very different for the given function. For such functions, preconditioning, which changes the geometry of the space to shape the function level sets like concentric circles, cures the slow convergence. Constructing and applying preconditioning can be computationally expensive, however.
The gradient descent can be modified via momentums (Nesterov, Polyak, and Frank–Wolfe) and heavy-ball parameters (exponential moving averages and positive-negative momentum). The main examples of such optimizers are Adam, DiffGrad, Yogi, AdaBelief, etc.
Methods based on Newton's method and inversion of the Hessian using conjugate gradient techniques can be better alternatives. Generally, such methods converge in fewer iterations, but the cost of each iteration is higher. An example is the BFGS method which consists in calculating on every step a matrix by which the gradient vector is multiplied to go into a "better" direction, combined with a more sophisticated line search algorithm, to find the "best" value of η. For extremely large problems, where the computer-memory issues dominate, a limited-memory method such as L-BFGS should be used instead of BFGS or the steepest descent. 
While it is sometimes possible to substitute gradient descent for a local search algorithm, gradient descent is not in the same family: although it is an iterative method for local optimization, it relies on an objective function's gradient rather than an explicit exploration of a solution space.
Gradient descent can be viewed as applying Euler's method for solving ordinary differential equations x′(t)=−∇f(x(t)) to a gradient flow.  In turn, this equation may be derived as an optimal controller for the control system x′(t)=u(t) with u(t) given in feedback form u(t)=−∇f(x(t)).


== Modifications ==
Gradient descent can converge to a local minimum and slow down in a neighborhood of a saddle point. Even for unconstrained quadratic minimization, gradient descent develops a zig–zag pattern of subsequent iterates as iterations progress, resulting in slow convergence. Multiple modifications of gradient descent have been proposed to address these deficiencies.


=== Fast gradient methods ===
Yurii Nesterov has proposed a simple modification that enables faster convergence for convex problems and has been since further generalized. For unconstrained smooth problems, the method is called the fast gradient method (FGM) or the accelerated gradient method (AGM). Specifically, if the differentiable function f is convex and ∇f is Lipschitz, and it is not assumed that f is strongly convex, then the error in the objective value generated at each step k by the gradient descent method will be bounded by 𝒪(k−1). Using the Nesterov acceleration technique, the error decreases at 𝒪(k−2). It is known that the rate 𝒪(k−2) for the decrease of the cost function is optimal for first-order optimization methods. Nevertheless, there is the opportunity to improve the algorithm by reducing the constant factor. The optimized gradient method (OGM) reduces that constant by a factor of two and is an optimal first-order method for large-scale problems.
For constrained or non-smooth problems, Nesterov's FGM is called the fast proximal gradient method (FPGM), an acceleration of the proximal gradient method.


=== Momentum or heavy ball method ===
Trying to break the zig-zag pattern of gradient descent, the momentum or heavy ball method uses a momentum term in analogy to a heavy ball sliding on the surface of values of the function being minimized, or to mass movement in Newtonian dynamics through a viscous medium in a conservative force field. Gradient descent with momentum remembers the solution update at each iteration, and determines the next update as a linear combination of the gradient and the previous update. For unconstrained quadratic minimization, a theoretical convergence rate bound of the heavy ball method is asymptotically the same as that for the optimal conjugate gradient method.
This technique is used in stochastic gradient descent and as an extension to the backpropagation algorithms used to train artificial neural networks. In the direction of updating, stochastic gradient descent adds a stochastic property. The weights can be used to calculate the derivatives.


== Extensions ==
Gradient descent can be extended to handle constraints by including a projection onto the set of constraints. This method is only feasible when the projection is efficiently computable on a computer. Under suitable assumptions, this method converges.  This method is a specific case of the forward–backward algorithm for monotone inclusions (which includes convex programming and variational inequalities).
Gradient descent is a special case of mirror descent using the squared Euclidean distance as the given Bregman divergence.


== Theoretical properties ==
The properties of gradient descent depend on the properties of the objective function and the variant of gradient descent used (for example, if a line search step is used). The assumptions made affect the convergence rate, and other properties, that can be proven for gradient descent. For example, if the objective is assumed to be strongly convex and lipschitz smooth, then gradient descent converges linearly with a fixed step size. Looser assumptions lead to either weaker convergence guarantees or require a more sophisticated step size selection.


== Examples ==
Yang–Mills flow
Yang–Mills–Higgs flow
Seiberg–Witten flow


== See also ==


== References ==


== Further reading ==
Boyd, Stephen; Vandenberghe, Lieven (2004). "Unconstrained Minimization" (PDF). Convex Optimization. New York: Cambridge University Press. pp. 457–520. ISBN 0-521-83378-7.
Chong, Edwin K. P.; Żak, Stanislaw H. (2013). "Gradient Methods". An Introduction to Optimization (Fourth ed.). Hoboken: Wiley. pp. 131–160. ISBN 978-1-118-27901-4.
Himmelblau, David M. (1972). "Unconstrained Minimization Procedures Using Derivatives". Applied Nonlinear Programming. New York: McGraw-Hill. pp. 63–132. ISBN 0-07-028921-2.


== External links ==

Using gradient descent in C++, Boost, Ublas for linear regression
Series of Khan Academy videos discusses gradient ascent
Online book teaching gradient descent in deep neural network context
Archived at Ghostarchive and the Wayback Machine: "Gradient Descent, How Neural Networks Learn". 3Blue1Brown. October 16, 2017 – via YouTube.
Garrigos, Guillaume; Gower, Robert M. (2023). "Handbook of Convergence Theorems for (Stochastic) Gradient Methods". arXiv:2301.11235 [math.OC].