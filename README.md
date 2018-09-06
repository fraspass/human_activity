# Separating human and automated activity in computer network traffic data

This reposit contains **python** code used to separate human and automated activity on a single edge within a computer network. 

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level <img src="https://rawgit.com/fraspass/human_activity/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> according to the the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270625pt height=14.15535pt/>-value obtained from a Fourier's <img src="https://rawgit.com/fraspass/human_activity/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Transforming the raw arrival times

Assume that <img src="https://rawgit.com/fraspass/human_activity/master/svgs/e2c473b0627500251619ee3222b5f1ba.svg?invert_in_darkmode" align=middle width=67.4223pt height=20.22207pt/> are the raw arrival times of events on an edge <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/>, where <img src="https://rawgit.com/fraspass/human_activity/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.90874pt height=22.46574pt/> and <img src="https://rawgit.com/fraspass/human_activity/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.19637pt height=22.46574pt/> are client and server IP addresses. Also assume that the edge <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0fa0326f423a749421f358bd1d3a1653.svg?invert_in_darkmode" align=middle width=53.675655pt height=22.46574pt/> is strongly periodic with periodicity <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270625pt height=14.15535pt/>, detected using the Fourier's <img src="https://rawgit.com/fraspass/human_activity/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.43051pt height=14.15535pt/>-test. The indicator variable <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/> takes the value <img src="https://rawgit.com/fraspass/human_activity/master/svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if the event associated with the arrival time <img src="https://rawgit.com/fraspass/human_activity/master/svgs/02ab12d0013b89c8edc7f0f2662fa7a9.svg?invert_in_darkmode" align=middle width=10.58706pt height=20.22207pt/> is automated, <img src="https://rawgit.com/fraspass/human_activity/master/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219277pt height=21.18732pt/> if it is human. Two quantities are used to make inference on <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/>:

<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/702ad402b3a1f9048139b7486d543baa.svg?invert_in_darkmode" align=middle width=472.9659pt height=36.18648pt/></p>

where <img src="https://rawgit.com/fraspass/human_activity/master/svgs/4bda6e2d17a6dd8e156052e83dde1de1.svg?invert_in_darkmode" align=middle width=41.096055pt height=21.18732pt/> is the number of seconds in one day. The <img src="https://rawgit.com/fraspass/human_activity/master/svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width=14.045955pt height=14.15535pt/> 's are **wrapped arrival times**, and the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710445pt height=14.15535pt/> 's are **daily arrival times**. 

### Mixture modelling

The following mixture model is used to make inference on the <img src="https://rawgit.com/fraspass/human_activity/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.295635pt height=14.15535pt/>'s:
<p align="center"><img src="https://rawgit.com/fraspass/human_activity/master/svgs/b8a19659f31c92039e9b5da0a4e3b39d.svg?invert_in_darkmode" align=middle width=206.65755pt height=18.31236pt/></p>

The distribution of <img src="https://rawgit.com/fraspass/human_activity/master/svgs/a5db2864f408f1246504f17cd9c63105.svg?invert_in_darkmode" align=middle width=36.107445pt height=24.6576pt/> is chosen to be **wrapped normal**, and for <img src="https://rawgit.com/fraspass/human_activity/master/svgs/04a94bf0af1c46c432a53d344a452748.svg?invert_in_darkmode" align=middle width=37.86783pt height=24.6576pt/>, a *step function* with unknown number <img src="https://rawgit.com/fraspass/human_activity/master/svgs/d30a65b936d8007addc9c789d5a7ae49.svg?invert_in_darkmode" align=middle width=6.8494305pt height=22.83138pt/> of changepoints <img src="https://rawgit.com/fraspass/human_activity/master/svgs/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode" align=middle width=9.04695pt height=14.15535pt/> is used. Conjugate priors are used for efficient implementation. In the code, a Collapsed Metropolis-within-Gibbs with Reversible Jump steps is used. 

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
