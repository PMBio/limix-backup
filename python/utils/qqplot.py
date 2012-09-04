def qqplot(M,alphaLevel = 0.05,logSpace = True):
    import scipy as SP
    import pylab as PL
    import scipy.stats as ST
    mRange=SP.unique(SP.round(SP.exp(SP.log10(1):0.1:SP.log10(M))));
    numPts=length(mRange);
    betaalphaLevel=SP.zeros((1,numPts));#down in the plot
    betaOneMinusalphaLevel=SP.zeros((1,numPts));#up in the plot
    betaInvHalf=SP.zeros((1,numPts));
    for n in SP.arange(numPts):
        m=mRange(n);
        numPLessThanThresh=m;
        betaInvHalf(n)=ST.beta.ppf(0.5,m,M-m);
        betaalphaLevel(n)=ST.beta.ppf(alphaLevel,m,M-m);
        betaOneMinusalphaLevel(n)=ST.beta.ppf(1-alphaLevel,m,M-m);
        pass
    betaDown=betaInvHalf-betaalphaLevel;
    betaUp=betaOneMinusalphaLevel-betaInvHalf;

    theoreticalPvals=mRange/M;
    PL.figure()
    PL.plot(-SP.log10(theoreticalPvals),-SP.log10(theoreticalPvals + betaUp));
    PL.plot(-SP.log10(theoreticalPvals),-SP.log10(theoreticalPvals - betaDown);
    PL.show()
