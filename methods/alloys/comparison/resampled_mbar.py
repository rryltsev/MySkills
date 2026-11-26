"""Newton MBAR for resampled histograms, initialized at the full-data solution."""
import numpy as np
from scipy.special import logsumexp


class ResampledMBAR:
    def __init__(self,k,q,specs,kt,reference,offset,hist):
        bias=(specs[:,1,None]*(q-specs[:,2,None])**2/2-(specs[:,0,None]-reference)*k)/kt
        self.counts=hist.sum(axis=1)
        terms=np.log(self.counts)[:,None]+offset[:,None]-bias
        self.logden=logsumexp(terms,axis=0)
        self.post=np.exp(terms-self.logden)

    def fit(self,hist):
        counts=hist.sum(axis=1)
        h=hist.sum(axis=0)
        total=h.sum()
        coeff=counts/self.counts
        delta=np.zeros(len(counts))
        def measure(d):
            shift=d.max()
            factor=coeff*np.exp(d-shift)
            den=factor@self.post
            return factor,den,float(h@(np.log(den)+shift)-counts@d)
        for _ in range(24):
            factor,den,val=measure(delta)
            p=(factor[:,None]*self.post)/den
            mean=p@h
            gradient=mean-counts
            if np.max(abs(gradient))/total<2e-10:
                base=np.full(len(h),-np.inf)
                sel=h>0
                base[sel]=np.log(h[sel])-self.logden[sel]-np.log(den[sel])-delta.max()
                return base
            # Gauge the first offset to zero. Off-diagonal Hessian terms
            # retain the complete overlap of the biased ensembles.
            rhs=gradient[1:]
            proj=p[1:]*np.sqrt(h)
            Hess=np.diag(mean[1:])-proj@proj.T
            step=np.linalg.solve(Hess,rhs)
            step=np.r_[0,step]
            predicted=float(gradient@step)
            if predicted<=0:
                raise ArithmeticError("Nonpositive curvature in MBAR resample")
            for power in range(20):
                scale=2.**(-power)
                trial=delta-scale*step
                _,d,v=measure(trial)
                if np.all(d>0) and v<=val-1e-4*scale*predicted+2.e-10:
                    delta=trial
                    break
            else:
                raise ArithmeticError("Nonconvergent line search in MBAR resample")
        raise ArithmeticError("MBAR bootstrap Newton updates did not converge")



if __name__ == "__main__":
    # Synthetic three-state system: two connected order umbrellas and known
    # unbiased probabilities at the reference chemical potential.
    k = np.array([0., 1., 2.])
    q = k / 2
    kt = .07
    prior = np.array([.2, .5, .3])
    specs = np.array([[0., .10, .2], [.03, .15, .8]])
    bias = (specs[:, 1, None] * (q - specs[:, 2, None])**2 / 2 -
            specs[:, 0, None] * k) / kt
    weights = prior * np.exp(-bias)
    hist = 1000 * weights / weights.sum(axis=1, keepdims=True)
    solver = ResampledMBAR(k, q, specs, kt, 0., np.zeros(2), hist)
    recovered = solver.fit(hist)
    recovered = np.exp(recovered - logsumexp(recovered))
    np.testing.assert_allclose(recovered, prior, atol=1e-8)
    print("Recovered reference probabilities:", recovered)
