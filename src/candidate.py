## candidate.py는 reco muon과 truth muon의 매칭을 확인하는 코드입니다.
## 실행 명령은 python candidate.py --sample <sample_name> --n-events <number_of_events> 입니다.
import awkward as ak
import numpy as np


def build_opposite_charge_pairs(muons):
    """
    Build all opposite-charge muon pairs in each event.

    muons must contain:
    - pt
    - eta
    - phi
    - mass
    - charge
    """
    pairs = ak.combinations(muons, 2, fields=["mu1", "mu2"])

    opposite = pairs.mu1.charge * pairs.mu2.charge < 0

    return pairs[opposite]


def compute_mumu_mass(pairs):
    """
    Compute invariant mass of muon pairs.
    """

    pt1 = pairs.mu1.pt
    eta1 = pairs.mu1.eta
    phi1 = pairs.mu1.phi
    m1 = pairs.mu1.mass

    pt2 = pairs.mu2.pt
    eta2 = pairs.mu2.eta
    phi2 = pairs.mu2.phi
    m2 = pairs.mu2.mass

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass2 = e**2 - px**2 - py**2 - pz**2
    mass2 = ak.where(mass2 > 0, mass2, 0)

    return np.sqrt(mass2)