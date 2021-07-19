# Classification of periodic arrivals in event time data for filtering computer network traffic

This repository contains *python* code used to separate human and automated activity on a single edge within a computer network. The methodology is described in _Sanna Passino, F. and Heard, N. A., "Classification of periodic arrivals in event time data for filtering computer network traffic", Statistics and Computing 30(5), 1241–1254 (2020)_ ([link to the paper](https://link.springer.com/article/10.1007/s11222-020-09943-9)).

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> according to the the <img src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270624999999999pt height=14.15535pt/>-value obtained from a Fourier's <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Fourier analysis

Assume that <img src="svgs/e2c473b0627500251619ee3222b5f1ba.svg?invert_in_darkmode" align=middle width=67.42229999999999pt height=20.22207pt/> are the raw arrival times of events on an edge <img src="svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/>, where <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.90874pt height=22.46574pt/> and <img src="svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.196369999999998pt height=22.46574pt/> are client and server IP addresses. In NetFlow data, the <img src="svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg?invert_in_darkmode" align=middle width=10.58706pt height=20.22207pt/>'s are expressed in seconds from the Unix epoch (January 1, 1970). From the raw arrival times, the counting process <img src="svgs/bc26136196e30407c1303ffbe073b500.svg?invert_in_darkmode" align=middle width=33.7214988pt height=24.657534pt/> counts the number of events up to time <img src="svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999999pt height=20.221802699999998pt/> from the beginning of the observation period. 

Given <img src="svgs/de3e1f364fdb63b83b40dccd545f87da.svg?invert_in_darkmode" align=middle width=162.96200249999998pt height=24.657534pt/>, where <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999998pt height=22.465723499999996pt/> is the total observation time, the periodogram <img src="svgs/ec6220dd5b1b0b041c2cef5a4282e12f.svg?invert_in_darkmode" align=middle width=51.5026083pt height=31.141535699999995pt/> can be calculated as follows:
<p align="center"><img src="svgs/82d1c12d422abbbaf8fe5b36692b4478.svg?invert_in_darkmode" align=middle width=393.14140469999995pt height=53.95471455pt/></p>

The periodogram can be easily evaluated at the Fourier frequencies <img src="svgs/3b3c77a3ac66bae7c20ee3ae896d4007.svg?invert_in_darkmode" align=middle width=197.3341491pt height=24.657534pt/> using the Fast Fourier Transform (FFT). The presence of periodicities can be assessed using the Fourier's <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-statistic:
<p align="center"><img src="svgs/7610c527b41a3ca4260038f78e3cced2.svg?invert_in_darkmode" align=middle width=193.18586925pt height=49.3970961pt/></p>

The <img src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270624999999999pt height=14.15535pt/>-value associated with an observed value <img src="svgs/edd8e1b4a643e8dd7ef5f1c2b1cbdc59.svg?invert_in_darkmode" align=middle width=15.165552599999998pt height=22.638462pt/> of the test statistic for the null hypothesis <img src="svgs/30074edb23bec8e7c47c584ff885e5b5.svg?invert_in_darkmode" align=middle width=20.216950049999998pt height=22.465723499999996pt/> of no periodicities is:
<p align="center"><img src="svgs/a4d956ecfb6d2cf7f2cb0f06710456ad.svg?invert_in_darkmode" align=middle width=558.9765554999999pt height=52.38100065pt/></p>

where <img src="svgs/e08b72ddd2f1f1f6611e392a2f496078.svg?invert_in_darkmode" align=middle width=78.8337363pt height=24.657534pt/>.

### Transforming the raw arrival times

Assume that the edge <img src="svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/> is strongly periodic with periodicity <img src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270624999999999pt height=14.15535pt/>, detected using the Fourier's <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. The indicator variable <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/> takes the value <img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if the event associated with the arrival time <img src="svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg?invert_in_darkmode" align=middle width=10.58706pt height=20.22207pt/> is automated, <img src="svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if it is human. Two quantities are used to make inference on <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/>:

<p align="center"><img src="svgs/d4ed676d0d5d70216ae80838ba5eb5f9.svg?invert_in_darkmode" align=middle width=472.9659pt height=36.186479999999996pt/></p>

where <img src="svgs/4bda6e2d17a6dd8e156052e83dde1de1.svg?invert_in_darkmode" align=middle width=41.096055pt height=21.18732pt/> is the number of seconds in one day. The <img src="svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width=14.045955000000001pt height=14.15535pt/>'s are **wrapped arrival times**, and the <img src="svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710444999999998pt height=14.15535pt/>'s are **daily arrival times**. 

### Mixture modelling

The following mixture model is used to make inference on the <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/>'s:
<p align="center"><img src="svgs/40d4c0eac68b1ecd5ef441f523b878f4.svg?invert_in_darkmode" align=middle width=206.65755pt height=18.312359999999998pt/></p>

The distribution of <img src="svgs/a5db2864f408f1246504f17cd9c63105.svg?invert_in_darkmode" align=middle width=36.107445pt height=24.6576pt/> is chosen to be **wrapped normal**, and for <img src="svgs/04a94bf0af1c46c432a53d344a452748.svg?invert_in_darkmode" align=middle width=37.86783pt height=24.6576pt/>, a **step function** with unknown number <img src="svgs/d30a65b936d8007addc9c789d5a7ae49.svg?invert_in_darkmode" align=middle width=6.849430499999999pt height=22.83138pt/> of changepoints <img src="svgs/61cc5c68794cf7506b09230dec69d5d2.svg?invert_in_darkmode" align=middle width=63.77843999999999pt height=14.15535pt/> is used. The density of the wrapped normal distribution is:
<p align="center"><img src="svgs/4af4609163620931f8786c5e77f96051.svg?invert_in_darkmode" align=middle width=306.17235pt height=46.644015pt/></p>
The circular step function density for the human events is:
<p align="center"><img src="svgs/4a841a7d61dae1512379e53401223cbd.svg?invert_in_darkmode" align=middle width=386.16105pt height=50.171385pt/></p>

In the code, a Collapsed Metropolis-within-Gibbs with Reversible Jump steps is used. Conjugate priors are used for efficient implementation.

The model is summarised in the following picture:

![image_test](images/model_graphical.png)

Inference for the wrapped normal part is simple: the prior for <img src="svgs/9d11042b56fedc8436e0a185245a816f.svg?invert_in_darkmode" align=middle width=47.35368pt height=26.76201pt/> is <img src="svgs/4b7a504322031c7e23764e9b32eec8b3.svg?invert_in_darkmode" align=middle width=134.673pt height=24.6576pt/>, Normal Inverse Gamma, i.e. <img src="svgs/3df9d5c3fa64b9b6c933e846656f2353.svg?invert_in_darkmode" align=middle width=201.667455pt height=26.76201pt/>. Given sampled values of <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/> and <img src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/>, with <img src="svgs/0aea3200024b1acf230a433179a7b699.svg?invert_in_darkmode" align=middle width=97.00349999999999pt height=32.25617999999999pt/>,  the conditional posterior is conjugate with the following updated parameters:
<p align="center"><img src="svgs/d3511bd74e42e7b2b9e95f808fdf8fe7.svg?invert_in_darkmode" align=middle width=453.80609999999996pt height=168.6069pt/></p>

Inference for the step function for the human density uses Reversible Jump Markov Chain Monte Carlo with standard birth-death moves. The sampler heavily uses the following marginalised density:
<p align="center"><img src="svgs/be4658518c4d2e5c18e41f5eb01d6d4d.svg?invert_in_darkmode" align=middle width=750.3078pt height=51.46811999999999pt/></p>

where <img src="svgs/28ff91fb7571bd737f919050404240bd.svg?invert_in_darkmode" align=middle width=215.253555pt height=24.6576pt/> and <img src="svgs/787f83b9ff8c6a507c6aa738f41f1d97.svg?invert_in_darkmode" align=middle width=205.251255pt height=32.25617999999999pt/>. 

## Understanding the code

The periodicities can be calculated as follows:
```
cat Datasets/outlook.txt | ./fourier_test.py
```

The main part of the code is contained in the file `collapsed_gibbs.py`. The code in `mix_wrapped.py` is used to initialise the algorithm using a uniform - wrapped normal mixture fitted using the EM algorithm. Finally, `cps_circle.py` contains details about the proposals and utility functions used for the Reversible Jump steps for the step function density of the human component in the Gibbs sampler. For details about the periodicity detection procedure and relevant code, see the repository `fraspass/human_activity_julia`.

**- Important -** All the parameters in the code have the same names used in the paper, except `z[i]`, which does not correspond to <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/>, but combines the latent variables <img src="svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295634999999999pt height=14.15535pt/> and <img src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/> used in the paper. In the code, for a given positive integer <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18733pt height=22.46574pt/>, `z[i]`<img src="svgs/54d41642682aacb167fcee29439be9e2.svg?invert_in_darkmode" align=middle width=150.456405pt height=24.6576pt/>. When `z[i]`<img src="svgs/e2e5e2fce3793186cee0c21e2c25bdbb.svg?invert_in_darkmode" align=middle width=56.84926499999999pt height=22.46574pt/>, then the event is classified as *human* (<img src="svgs/ebc835e29cd47503f744073a06507e62.svg?invert_in_darkmode" align=middle width=43.254419999999996pt height=21.18732pt/>), and when `z[i]`<img src="svgs/a9470acfe48181b1808e6259e2da97b7.svg?invert_in_darkmode" align=middle width=56.84926499999999pt height=22.83138pt/>, then the event is *automated* (<img src="svgs/828ee044f61b3e43491ac27de061a056.svg?invert_in_darkmode" align=middle width=43.254419999999996pt height=21.18732pt/>), and the value represents a sample for <img src="svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/>, truncated to <img src="svgs/e6273ada5689f427d225a0cef5436d30.svg?invert_in_darkmode" align=middle width=88.12782pt height=24.6576pt/> for a suitably large <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18733pt height=22.46574pt/>. 

## Running the code

From a sequence <img src="svgs/e2c473b0627500251619ee3222b5f1ba.svg?invert_in_darkmode" align=middle width=67.42229999999999pt height=20.22207pt/> of event times, contained in a file `example_data.txt` (one arrival time per line), the code can be run as follows:
```
cat example_data.txt | ./filter_human.py
```

Several arguments can be passed to `filter_human.py`. Calling `./filter_human.py --help` returns detailed instruction on the possible options:
```
$ ./filter_human.py --help
usage: filter_human.py [-h] [-N NSAMP] [-B NBURN] [-C NCHAIN] [-p PERIOD] [-r]
                       [-o] [-l] [-s] [-m MU] [-t TAU] [-a ALPHA] [-b BETA]
                       [-g GAMMA] [-d DELTA] [-e ETA] [-v NU] [-k LMAX]
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
  -r, --trunc           Round the periodicity to 2 decimal digits, default
                        False.
  -o, --fixed_duration  Add if the fixed duration model should be used,
                        default False.
  -l, --laplace         Add if the wrapped Laplace model should be used,
                        default False.
  -s, --shift           Add if the wrapped events should be shifted by pi,
                        default False.
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
  -k LMAX, --lmax LMAX  Integer: maximum (absolute) value for kappa, default 5
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
