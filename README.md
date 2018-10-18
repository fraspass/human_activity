# Statistical methods for separating human and automated activity in computer network traffic

This reposit contains *python* code used to separate human and automated activity on a single edge within a computer network. 

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level <img src="https://rawgit.com/fraspass/human_activity/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> according to the the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270625pt height=14.15535pt/>-value obtained from a Fourier's <img src="https://rawgit.com/fraspass/human_activity/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Transforming the raw arrival times

Assume that <img src="https://rawgit.com/fraspass/human_activity/master/svgs/e2c473b0627500251619ee3222b5f1ba.svg?invert_in_darkmode" align=middle width=67.4223pt height=20.22207pt/> are the raw arrival times of events on an edge <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/>, where <img src="https://rawgit.com/fraspass/human_activity/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.90874pt height=22.46574pt/> and <img src="https://rawgit.com/fraspass/human_activity/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.19637pt height=22.46574pt/> are client and server IP addresses. Also assume that the edge <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/> is strongly periodic with periodicity <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270625pt height=14.15535pt/>, detected using the Fourier's <img src="https://rawgit.com/fraspass/human_activity/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. The indicator variable <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/> takes the value <img src="https://rawgit.com/fraspass/human_activity/master/svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if the event associated with the arrival time <img src="https://rawgit.com/fraspass/human_activity/master/svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg?invert_in_darkmode" align=middle width=10.58706pt height=20.22207pt/> is automated, <img src="https://rawgit.com/fraspass/human_activity/master/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if it is human. Two quantities are used to make inference on <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/>:

<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/d4ed676d0d5d70216ae80838ba5eb5f9.svg?invert_in_darkmode" align=middle width=472.9659pt height=36.18648pt/></p>

where <img src="https://rawgit.com/fraspass/human_activity/master/svgs/4bda6e2d17a6dd8e156052e83dde1de1.svg?invert_in_darkmode" align=middle width=41.096055pt height=21.18732pt/> is the number of seconds in one day. The <img src="https://rawgit.com/fraspass/human_activity/master/svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width=14.045955pt height=14.15535pt/>'s are **wrapped arrival times**, and the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710445pt height=14.15535pt/>'s are **daily arrival times**. 

### Mixture modelling

The following mixture model is used to make inference on the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/>'s:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/40d4c0eac68b1ecd5ef441f523b878f4.svg?invert_in_darkmode" align=middle width=206.65755pt height=18.31236pt/></p>

The distribution of <img src="https://rawgit.com/fraspass/human_activity/master/svgs/a5db2864f408f1246504f17cd9c63105.svg?invert_in_darkmode" align=middle width=36.107445pt height=24.6576pt/> is chosen to be **wrapped normal**, and for <img src="https://rawgit.com/fraspass/human_activity/master/svgs/04a94bf0af1c46c432a53d344a452748.svg?invert_in_darkmode" align=middle width=37.86783pt height=24.6576pt/>, a **step function** with unknown number <img src="https://rawgit.com/fraspass/human_activity/master/svgs/d30a65b936d8007addc9c789d5a7ae49.svg?invert_in_darkmode" align=middle width=6.8494305pt height=22.83138pt/> of changepoints <img src="https://rawgit.com/fraspass/human_activity/master/svgs/61cc5c68794cf7506b09230dec69d5d2.svg?invert_in_darkmode" align=middle width=63.77844pt height=14.15535pt/> is used. The density of the wrapped normal distribution is:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/4af4609163620931f8786c5e77f96051.svg?invert_in_darkmode" align=middle width=306.17235pt height=46.644015pt/></p>
The circular step function density for the human events is:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/c97d0f2e6c6ccc89bf0ecd311823b29d.svg?invert_in_darkmode" align=middle width=395.0661pt height=50.171385pt/></p>

In the code, a Collapsed Metropolis-within-Gibbs with Reversible Jump steps is used. Conjugate priors are used for efficient implementation.

The model is summarised in the following picture:

![image_test](images/model_graphical.png)

Inference for the wrapped normal part is simple: the prior for <img src="https://rawgit.com/fraspass/human_activity/master/svgs/9d11042b56fedc8436e0a185245a816f.svg?invert_in_darkmode" align=middle width=47.35368pt height=26.76201pt/> is <img src="https://rawgit.com/fraspass/human_activity/master/svgs/4b7a504322031c7e23764e9b32eec8b3.svg?invert_in_darkmode" align=middle width=134.673pt height=24.6576pt/>, Normal Inverse Gamma, i.e. <img src="https://rawgit.com/fraspass/human_activity/master/svgs/3df9d5c3fa64b9b6c933e846656f2353.svg?invert_in_darkmode" align=middle width=201.667455pt height=26.76201pt/>. Given sampled values of <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/> and <img src="https://rawgit.com/fraspass/human_activity/master/svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/>, with <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0aea3200024b1acf230a433179a7b699.svg?invert_in_darkmode" align=middle width=97.0035pt height=32.25618pt/>,  the conditional posterior is conjugate with the following updated parameters:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/d3511bd74e42e7b2b9e95f808fdf8fe7.svg?invert_in_darkmode" align=middle width=453.8061pt height=168.6069pt/></p>

Inference for the step function for the human density uses Reversible Jump Markov Chain Monte Carlo with standard birth-death moves. The sampler heavily uses the following marginalised density:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/be4658518c4d2e5c18e41f5eb01d6d4d.svg?invert_in_darkmode" align=middle width=750.3078pt height=51.46812pt/></p>

where <img src="https://rawgit.com/fraspass/human_activity/master/svgs/28ff91fb7571bd737f919050404240bd.svg?invert_in_darkmode" align=middle width=215.253555pt height=24.6576pt/> and <img src="https://rawgit.com/fraspass/human_activity/master/svgs/787f83b9ff8c6a507c6aa738f41f1d97.svg?invert_in_darkmode" align=middle width=205.251255pt height=32.25618pt/>. 


## Understanding the code

The main part of the code is contained in the file `collapsed_gibbs.py`. The code in `mix_wrapped.py` is used to initialise the algorithm using a uniform - wrapped normal mixture fitted using the EM algorithm. Finally, `cps_circle.py` contains details about the proposals and utility functions used for the Reversible Jump steps for the step function density of the human component in the Gibbs sampler. For details about the periodicity detection procedure and relevant code, see the repository `fraspass/human_activity_julia`.

**- Important -** All the parameters in the code have the same names used in the paper, except `z[i]`, which does not correspond to <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/>, but combines the latent variables <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/> and <img src="https://rawgit.com/fraspass/human_activity/master/svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/> used in the paper. In the code, for a given positive integer <img src="https://rawgit.com/fraspass/human_activity/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18733pt height=22.46574pt/>, `z[i]`<img src="https://rawgit.com/fraspass/human_activity/master/svgs/54d41642682aacb167fcee29439be9e2.svg?invert_in_darkmode" align=middle width=150.456405pt height=24.6576pt/>. When `z[i]`<img src="https://rawgit.com/fraspass/human_activity/master/svgs/e2e5e2fce3793186cee0c21e2c25bdbb.svg?invert_in_darkmode" align=middle width=56.849265pt height=22.46574pt/>, then the event is classified as *human* (<img src="https://rawgit.com/fraspass/human_activity/master/svgs/ebc835e29cd47503f744073a06507e62.svg?invert_in_darkmode" align=middle width=43.25442pt height=21.18732pt/>), and when `z[i]`<img src="https://rawgit.com/fraspass/human_activity/master/svgs/a9470acfe48181b1808e6259e2da97b7.svg?invert_in_darkmode" align=middle width=56.849265pt height=22.83138pt/>, then the event is *automated* (<img src="https://rawgit.com/fraspass/human_activity/master/svgs/828ee044f61b3e43491ac27de061a056.svg?invert_in_darkmode" align=middle width=43.25442pt height=21.18732pt/>), and the value represents a sample for <img src="https://rawgit.com/fraspass/human_activity/master/svgs/061e7c3be0101eabfbaa013fe337ba95.svg?invert_in_darkmode" align=middle width=14.12202pt height=14.15535pt/>, truncated to <img src="https://rawgit.com/fraspass/human_activity/master/svgs/e6273ada5689f427d225a0cef5436d30.svg?invert_in_darkmode" align=middle width=88.12782pt height=24.6576pt/> for a suitably large <img src="https://rawgit.com/fraspass/human_activity/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18733pt height=22.46574pt/>. 

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
