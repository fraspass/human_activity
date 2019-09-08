# Statistical methods for filtering polling activity in computer network traffic

This reposit contains *python* code used to separate human and automated activity on a single edge within a computer network. 

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level <img alt="$\alpha$" src="svgs/c745b9b57c145ec5577b82542b2df546.svg" align="middle" width="10.5765pt" height="14.15535pt"/> according to the the <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" align="middle" width="8.270625pt" height="14.15535pt"/>-value obtained from a Fourier's <img alt="$g$" src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg" align="middle" width="8.43051pt" height="14.15535pt"/>-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Fourier analysis

Assume that <img alt="$t_1,\dots,t_N$" src="svgs/e2c473b0627500251619ee3222b5f1ba.svg" align="middle" width="67.4223pt" height="20.22207pt"/> are the raw arrival times of events on an edge <img alt="$X\to Y$" src="svgs/0fa0326f423a749421f358bd1d3a1653.svg" align="middle" width="53.675655pt" height="22.46574pt"/>, where <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.90874pt" height="22.46574pt"/> and <img alt="$Y$" src="svgs/91aac9730317276af725abd8cef04ca9.svg" align="middle" width="13.19637pt" height="22.46574pt"/> are client and server IP addresses. In NetFlow data, the <img alt="$t_i$" src="svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg" align="middle" width="10.58706pt" height="20.22207pt"/>'s are expressed in seconds from the Unix epoch (January 1, 1970). From the raw arrival times, the counting process <img alt="$N(t)$" src="svgs/bc26136196e30407c1303ffbe073b500.svg" align="middle" width="33.7214988pt" height="24.657534pt"/> counts the number of events up to time <img alt="$t$" src="svgs/4f4f4e395762a3af4575de74c019ebb5.svg" align="middle" width="5.93609775pt" height="20.2218027pt"/> from the beginning of the observation period. 

Given <img alt="$\{N(t),\ t=0,1,\dots,T\}$" src="svgs/de3e1f364fdb63b83b40dccd545f87da.svg" align="middle" width="162.9620025pt" height="24.657534pt"/>, where <img alt="$T$" src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg" align="middle" width="11.88931425pt" height="22.4657235pt"/> is the total observation time, the periodogram <img alt="$\hat{S}^{(p)}(f)$" src="svgs/ec6220dd5b1b0b041c2cef5a4282e12f.svg" align="middle" width="51.5026083pt" height="31.1415357pt"/> can be calculated as follows:
<p align="center"><img alt="\begin{equation*}&#10;\hat{S}^{(p)}(f) = \left\vert\frac{1}{T}\sum_{t=1}^T \left(N(t)-N(t-1)-\frac{N(T)}{T}\right)e^{-2\pi\imath ft}\right\vert^2&#10;\end{equation*}" src="svgs/82d1c12d422abbbaf8fe5b36692b4478.svg" align="middle" width="393.1414047pt" height="53.95471455pt"/></p>

The periodogram can be easily evaluated at the Fourier frequencies <img alt="$f_k = k/T,\ k=0,\dots,\lfloor T/2 \rfloor$" src="svgs/3b3c77a3ac66bae7c20ee3ae896d4007.svg" align="middle" width="197.3341491pt" height="24.657534pt"/> using the Fast Fourier Transform (FFT). The presence of periodicities can be assessed using the Fourier's <img alt="$g$" src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg" align="middle" width="8.43051pt" height="14.15535pt"/>-statistic:
<p align="center"><img alt="\begin{equation*}&#10;g = \frac{\max_{1\leq k\leq\lfloor T/2\rfloor} \hat{S}^{(p)}(f_k)}{\sum_{1\leq k\leq\lfloor T/2\rfloor} \hat{S}^{(p)}(f_k)}&#10;\end{equation*}" src="svgs/7610c527b41a3ca4260038f78e3cced2.svg" align="middle" width="193.18586925pt" height="49.3970961pt"/></p>

The <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" align="middle" width="8.270625pt" height="14.15535pt"/>-value associated with an observed value <img alt="$g^\star$" src="svgs/edd8e1b4a643e8dd7ef5f1c2b1cbdc59.svg" align="middle" width="15.1655526pt" height="22.638462pt"/> of the test statistic for the null hypothesis <img alt="$H_0$" src="svgs/30074edb23bec8e7c47c584ff885e5b5.svg" align="middle" width="20.21695005pt" height="22.4657235pt"/> of no periodicities is:
<p align="center"><img alt="\begin{equation*}&#10;\mathbb P(g&gt;g^\star) = \sum_{j=1}^{\min\{\lfloor 1/g^\star\rfloor,m\}} (-1)^{j-1}\binom{m}{j}(1-jg^\star)^{m-1} \approx 1-(1-\exp\{-mg^\star\} )^{m}&#10;\end{equation*}" src="svgs/a4d956ecfb6d2cf7f2cb0f06710456ad.svg" align="middle" width="558.9765555pt" height="52.38100065pt"/></p>

where <img alt="$m = \lfloor T/2\rfloor$" src="svgs/e08b72ddd2f1f1f6611e392a2f496078.svg" align="middle" width="78.8337363pt" height="24.657534pt"/>.

### Transforming the raw arrival times

Assume that the edge <img alt="$X\to Y$" src="svgs/0fa0326f423a749421f358bd1d3a1653.svg" align="middle" width="53.675655pt" height="22.46574pt"/> is strongly periodic with periodicity <img alt="$p$" src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" align="middle" width="8.270625pt" height="14.15535pt"/>, detected using the Fourier's <img alt="$g$" src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg" align="middle" width="8.43051pt" height="14.15535pt"/>-test. The indicator variable <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/> takes the value <img alt="$1$" src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg" align="middle" width="8.219277pt" height="21.18732pt"/> if the event associated with the arrival time <img alt="$t_i$" src="svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg" align="middle" width="10.58706pt" height="20.22207pt"/> is automated, <img alt="$0$" src="svgs/29632a9bf827ce0200454dd32fc3be82.svg" align="middle" width="8.219277pt" height="21.18732pt"/> if it is human. Two quantities are used to make inference on <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/>:

<p align="center"><img alt="\begin{align*}&#10;x_i=(t_i\ \text{mod}\ p){}\times \frac{2\pi}{p} &amp; &amp; y_i=(t_i\ \text{mod}\ 86400){}\times\frac{2\pi}{86400}&#10;\end{align*}" src="svgs/d4ed676d0d5d70216ae80838ba5eb5f9.svg" align="middle" width="472.9659pt" height="36.18648pt"/></p>

where <img alt="$86400$" src="svgs/4bda6e2d17a6dd8e156052e83dde1de1.svg" align="middle" width="41.096055pt" height="21.18732pt"/> is the number of seconds in one day. The <img alt="$x_i$" src="svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg" align="middle" width="14.045955pt" height="14.15535pt"/>'s are **wrapped arrival times**, and the <img alt="$y_i$" src="svgs/2b442e3e088d1b744730822d18e7aa21.svg" align="middle" width="12.710445pt" height="14.15535pt"/>'s are **daily arrival times**. 

### Mixture modelling

The following mixture model is used to make inference on the <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/>'s:
<p align="center"><img alt="\begin{equation*}&#10;f(t_i|z_i) \propto f_A(x_i)^{z_i} f_H(y_i)^{1-z_i} &#10;\end{equation*}" src="svgs/40d4c0eac68b1ecd5ef441f523b878f4.svg" align="middle" width="206.65755pt" height="18.31236pt"/></p>

The distribution of <img alt="$f_A(\cdot)$" src="svgs/a5db2864f408f1246504f17cd9c63105.svg" align="middle" width="36.107445pt" height="24.6576pt"/> is chosen to be **wrapped normal**, and for <img alt="$f_H(\cdot)$" src="svgs/04a94bf0af1c46c432a53d344a452748.svg" align="middle" width="37.86783pt" height="24.6576pt"/>, a **step function** with unknown number <img alt="$\ell$" src="svgs/d30a65b936d8007addc9c789d5a7ae49.svg" align="middle" width="6.8494305pt" height="22.83138pt"/> of changepoints <img alt="$\tau_1,\dots,\tau_\ell$" src="svgs/61cc5c68794cf7506b09230dec69d5d2.svg" align="middle" width="63.77844pt" height="14.15535pt"/> is used. The density of the wrapped normal distribution is:
<p align="center"><img alt="\begin{equation*}&#10;\phi_{\mathrm{WN}}^{[0,2\pi)}(x_i;\mu,\sigma^2)=\sum_{k=-\infty}^\infty \phi(x_i+2\pi k;\mu, \sigma^2)&#10;\end{equation*}" src="svgs/4af4609163620931f8786c5e77f96051.svg" align="middle" width="306.17235pt" height="46.644015pt"/></p>
The circular step function density for the human events is:
<p align="center"><img alt="\begin{equation*}&#10;h(y_i;\boldsymbol h,\boldsymbol \tau,\ell)=\frac{\mathbb{I}_{[0,\tau_{1})\cup[\tau_{\ell},2\pi)}(y_i) h_\ell}{2\pi-\tau_{\ell}+\tau_{1}}+\sum_{j=1}^{\ell-1} \frac{\mathbb{I}_{[\tau_{j},\tau_{j+1})}(y) h_j}{\tau_{j+1}-\tau_{j}}&#10;\end{equation*}" src="svgs/4a841a7d61dae1512379e53401223cbd.svg" align="middle" width="386.16105pt" height="50.171385pt"/></p>

In the code, a Collapsed Metropolis-within-Gibbs with Reversible Jump steps is used. Conjugate priors are used for efficient implementation.

The model is summarised in the following picture:

![image_test](images/model_graphical.png)

Inference for the wrapped normal part is simple: the prior for <img alt="$(\mu,\sigma^2)$" src="svgs/9d11042b56fedc8436e0a185245a816f.svg" align="middle" width="47.35368pt" height="26.76201pt"/> is <img alt="$\mathrm{NIG}(\mu_0,\lambda_0,\alpha_0,\beta_0)$" src="svgs/4b7a504322031c7e23764e9b32eec8b3.svg" align="middle" width="134.673pt" height="24.6576pt"/>, Normal Inverse Gamma, i.e. <img alt="$\mathrm{IG}(\sigma^2\vert\alpha_0,\beta_0) \mathbb{N}(\mu\vert\mu_0,\sigma^2/\lambda_0)$" src="svgs/3df9d5c3fa64b9b6c933e846656f2353.svg" align="middle" width="201.667455pt" height="26.76201pt"/>. Given sampled values of <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/> and <img alt="$\kappa_i$" src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg" align="middle" width="14.12202pt" height="14.15535pt"/>, with <img alt="$N_1=\sum_{i=1}^N z_i$" src="svgs/0aea3200024b1acf230a433179a7b699.svg" align="middle" width="97.0035pt" height="32.25618pt"/>,  the conditional posterior is conjugate with the following updated parameters:
<p align="center"><img alt="\begin{align*}&#10;\tilde{x} &amp;= \sum_{i:z_i=1}\nolimits (x_i+2\pi\kappa_i)/{N_1} \\&#10;\mu_{N_1} &amp;= \frac{\lambda_0\mu_0 + N_1\tilde{x}}{\lambda_0+N_1} \\&#10;\lambda_{N_1} &amp;= \lambda_0 + N_1 \\&#10;\alpha_{N_1} &amp;= \alpha_0 + N_1/2 \\&#10;\beta_{N_1} &amp;= \beta_0 + \frac{1}{2}\left\{\sum_{i:z_i=1}\nolimits (x_i+2\pi\kappa_i-\tilde{x})^2 + \frac{\lambda_0N_1}{\lambda_0+N_1}(\tilde x-\mu_0)^2 \right\}&#10;\end{align*}" src="svgs/d3511bd74e42e7b2b9e95f808fdf8fe7.svg" align="middle" width="453.8061pt" height="168.6069pt"/></p>

Inference for the step function for the human density uses Reversible Jump Markov Chain Monte Carlo with standard birth-death moves. The sampler heavily uses the following marginalised density:
<p align="center"><img alt="\begin{equation*}&#10;p(\boldsymbol{y}\vert\tau_1,\dots,\tau_\ell,\ell) = \frac{c(N,\eta)\Gamma[N-\sum_{j=1}^{\ell-1} N_{\tau_j,\tau_{j+1}}+\eta(2\pi-\tau_{\ell}+\tau_{1})]}{\Gamma[\eta(2\pi-\tau_{\ell}+\tau_{1})](2\pi-\tau_\ell+\tau_1)^{N-\sum_{h=1}^{\ell-1} N_{\tau_h,\tau_{h+1}}}} \prod_{j=1}^{\ell-1}\frac{\Gamma[N_{\tau_j,\tau_{j+1}}+\eta(\tau_{j+1}-\tau_{j})]}{\Gamma[\eta(\tau_{j+1}-\tau_{j})](\tau_{j+1}-\tau_j)^{N_{\tau_j,\tau_{j+1}}}}  &#10;\end{equation*}" src="svgs/be4658518c4d2e5c18e41f5eb01d6d4d.svg" align="middle" width="750.3078pt" height="51.46812pt"/></p>

where <img alt="$c(N,\eta)=\Gamma(2\pi\eta)/\Gamma(N+2\pi\eta)$" src="svgs/28ff91fb7571bd737f919050404240bd.svg" align="middle" width="215.253555pt" height="24.6576pt"/> and <img alt="$N_{\tau_{j},\tau_{j+1}} = \sum_{i=1}^N \mathbb{I}_{[\tau_{j},\tau_{j+1})}(y_i)$" src="svgs/787f83b9ff8c6a507c6aa738f41f1d97.svg" align="middle" width="205.251255pt" height="32.25618pt"/>. 

## Understanding the code

The periodicities can be calculated as follows:
```
cat Datasets/outlook.txt | ./fourier_test.py
```

The main part of the code is contained in the file `collapsed_gibbs.py`. The code in `mix_wrapped.py` is used to initialise the algorithm using a uniform - wrapped normal mixture fitted using the EM algorithm. Finally, `cps_circle.py` contains details about the proposals and utility functions used for the Reversible Jump steps for the step function density of the human component in the Gibbs sampler. For details about the periodicity detection procedure and relevant code, see the repository `fraspass/human_activity_julia`.

**- Important -** All the parameters in the code have the same names used in the paper, except `z[i]`, which does not correspond to <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/>, but combines the latent variables <img alt="$z_i$" src="svgs/6af8e9329c416994c3690752bde99a7d.svg" align="middle" width="12.295635pt" height="14.15535pt"/> and <img alt="$\kappa_i$" src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg" align="middle" width="14.12202pt" height="14.15535pt"/> used in the paper. In the code, for a given positive integer <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.18733pt" height="22.46574pt"/>, `z[i]`<img alt="$\in\{-L,\dots,L,L+1\}$" src="svgs/54d41642682aacb167fcee29439be9e2.svg" align="middle" width="150.456405pt" height="24.6576pt"/>. When `z[i]`<img alt="$=L+1$" src="svgs/e2e5e2fce3793186cee0c21e2c25bdbb.svg" align="middle" width="56.849265pt" height="22.46574pt"/>, then the event is classified as *human* (<img alt="$z_i=0$" src="svgs/ebc835e29cd47503f744073a06507e62.svg" align="middle" width="43.25442pt" height="21.18732pt"/>), and when `z[i]`<img alt="$\neq L+1$" src="svgs/a9470acfe48181b1808e6259e2da97b7.svg" align="middle" width="56.849265pt" height="22.83138pt"/>, then the event is *automated* (<img alt="$z_i=1$" src="svgs/828ee044f61b3e43491ac27de061a056.svg" align="middle" width="43.25442pt" height="21.18732pt"/>), and the value represents a sample for <img alt="$\kappa_i$" src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg" align="middle" width="14.12202pt" height="14.15535pt"/>, truncated to <img alt="$\{-L,\dots,L\}$" src="svgs/e6273ada5689f427d225a0cef5436d30.svg" align="middle" width="88.12782pt" height="24.6576pt"/> for a suitably large <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.18733pt" height="22.46574pt"/>. 

## Running the code

From a sequence <img alt="$t_1,\dots,t_N$" src="svgs/e2c473b0627500251619ee3222b5f1ba.svg" align="middle" width="67.4223pt" height="20.22207pt"/> of event times, contained in a file `example_data.txt` (one arrival time per line), the code can be run as follows:
```
cat example_data.txt | ./filter_human.py
```

Several arguments can be passed to `filter_human.py`. Calling `./filter_human.py --help` returns detailed instruction on the possible options:
```
usage: filter_human.py [-h] [-N NSAMP] [-B NBURN] [-C NCHAIN] [-p PERIOD]
                       [-l LMAX] [-m MU] [-t TAU] [-a ALPHA] [-b BETA]
                       [-g GAMMA] [-d DELTA] [-e ETA] [-v NU]
                       [-f [DEST_FOLDER]]

optional arguments:
  -h, --help            show this help message and exit
  -N NSAMP, --nsamp NSAMP
                        Integer: number of samples after burnin for each
                        chain, default 20000.
  -B NBURN, --nburn NBURN
                        Integer: number of samples after burnin for each
                        chain, default 5000.
  -C NCHAIN, --nchain NCHAIN
                        Integer: number of chains, default 1.
  -p PERIOD, --period PERIOD
                        Float: periodicity (if known), default: calculated via
                        Fourier test.
  -l LMAX, --lmax LMAX  Integer: maximum (absolute) value for kappa, default 5
  -m MU, --mu MU        Float: first parameter of the NIG prior, default pi.
  -t TAU, --tau TAU     Float: second parameter of the NIG prior, default 1.0.
  -a ALPHA, --alpha ALPHA
                        Float: third parameter of the NIG prior, default 1.0.
  -b BETA, --beta BETA  Float: fourth parameter of the NIG prior, default 1.0.
  -g GAMMA, --gamma GAMMA
                        Float: first parameter of the Beta prior on theta,
                        default 1.0.
  -d DELTA, --delta DELTA
                        Float: second parameter of the Beta prior on theta,
                        default 1.0.
  -e ETA, --eta ETA     Float: concentration parameter of the Dirichlet prior,
                        default 1.0.
  -v NU, --nu NU        Float: parameter of the Geometric prior, default 0.1.
  -f [DEST_FOLDER], --folder [DEST_FOLDER]
                        String: name of the destination folder for the output
                        files.
```

For example, the following call returns results obtained from 2 MCMC chains of length 2000, with burn-in 1000:
```
cat example_data.txt | ./filter_human.py -N 2000 -B 1000 -C 2
```

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
