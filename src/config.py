## config.py는 데이터셋의 경로, tree name, step size, feature list, histogram config 등을 정의하는 파일입니다.
SAMPLES = {
    "DYJetsToLL": {
        "url_pattern": "../dataset/DYJetsToLL_url*.txt",
        "data_dir": "../data/MC/DYJetsToLL",
    },
    "TTbar": {
        "url_pattern": "../dataset/TTbar_url*.txt",
        "data_dir": "../data/MC/TTbar",
    },
}

TREE_NAME = "Events"
STEP_SIZE = 100_000

CONTINUOUS_FEATURES = [
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_pfRelIso04_all",
    "Muon_dxy",
    "Muon_dz",
]

BOOLEAN_FEATURES = [
    "Muon_tightId",
]

FEATURES = CONTINUOUS_FEATURES + BOOLEAN_FEATURES

HIST_CONFIG = {
    "Muon_pt": {
        "range": (0, 200),
        "bins": 100,
        "xlabel": r"Muon $p_T$ [GeV]",
    },
    "Muon_eta": {
        "range": (-3, 3),
        "bins": 60,
        "xlabel": r"Muon $\eta$",
    },
    "Muon_phi": {
        "range": (-3.2, 3.2),
        "bins": 64,
        "xlabel": r"Muon $\phi$",
    },
    "Muon_pfRelIso04_all": {
        "range": (0, 1),
        "bins": 100,
        "xlabel": "Muon PF relative isolation",
    },
    "Muon_dxy": {
        "range": (-0.1, 0.1),
        "bins": 100,
        "xlabel": r"Muon $d_{xy}$ [cm]",
    },
    "Muon_dz": {
        "range": (-0.5, 0.5),
        "bins": 100,
        "xlabel": r"Muon $d_z$ [cm]",
    },
}