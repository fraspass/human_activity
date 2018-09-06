# Separating human and automated activity in computer network traffic data

This reposit contains **python** code used to separate human and automated activity on a single edge within a computer network. 

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level $\alpha$ according to the the $p$-value obtained from a Fourier's $g$-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Transforming the raw arrival times

Assume that $t_1,\dots,t_N$ are the raw arrival times of events on an edge $X\to Y$, where $X$ and $Y$ are client and server IP addresses. Also assume that the edge $X\to Y$ is strongly periodic with periodicity $p$, detected using the Fourier's $g$-test. The indicator variable $z_i$ takes the value $1$ if the event associated with the arrival time $t_i$ is automated, $0$ if it is human. Two quantities are used to make inference on $z_i$:

\begin{align*}
x_i=(t_i\ \text{mod}\ p){}\times\frac{2\pi}{p} & & y_i=(t_i\ \text{mod}\ 86400){}\times\frac{2\pi}{86400}
\end{align*}

where $86400$ is the number of seconds in one day. The $x_i$ 's are **wrapped arrival times**, and the $y_i$ 's are **daily arrival times**. 

### Mixture modelling

The following mixture model is used to make inference on the $z_i$'s:
\begin{equation*}
f(t_i|z_i)\propto f_A(x_i)^{z_i} f_H(y_i)^{1-z_i}
\end{equation*}

The distribution of $f_A(\cdot)$ is chosen to be *wrapped normal*, and for $f_H(\cdot)$, a *step function* with unknown number $\ell$ of changepoints $\tau$ is used. Conjugate priors are used for efficient implementation. In the code, a Collapsed Metropolis-within-Gibbs with Reversible Jump steps is used. 

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))