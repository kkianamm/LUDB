"""
LUDB 8-class (`ludb8`) config additions.

ONLY needed if you run `prepare_data_ludb.py --labelset ludb8`. For the default
`--labelset super5`, the repo's existing config_ipl.py already matches and you
change nothing.

How to use: paste the three blocks below into `config_ipl.py`, replacing the
existing CLASSNAMES / CLASS_PROMPTS and ADDING LABEL_ALIASES. Order must be
identical across all three (it defines the target-column order everywhere:
IPL prompts, data_fix target resolution, and the metrics table).

`data_fix.resolve_targets` matches each class against LABEL_ALIASES; the first
token in every alias row is the lowercased per-class column name written by
prepare_data_ludb.py, so target resolution hits "Path A" (one 0/1 column per
class) cleanly and `_guard` passes.
"""

# ---- 1) replace CLASSNAMES ------------------------------------------------
CLASSNAMES = [
    "normal ecg",
    "atrial fibrillation or flutter",
    "conduction disturbance",
    "atrial or ventricular extrasystoles",
    "cardiac hypertrophy",
    "myocardial ischemia or infarction",
    "st t wave change",
    "cardiac pacing",
]

# ---- 2) replace CLASS_PROMPTS (ensembled anchors for the Phase-B loss) -----
CLASS_PROMPTS = [
    ["a twelve lead ecg showing a normal heart rhythm",
     "an electrocardiogram with no abnormality",
     "normal ecg"],
    ["a twelve lead ecg showing atrial fibrillation",
     "an electrocardiogram showing atrial flutter",
     "ecg with an irregular non-sinus atrial rhythm"],
    ["a twelve lead ecg showing a conduction disturbance",
     "an electrocardiogram with a bundle branch block or av block",
     "ecg with conduction abnormality"],
    ["a twelve lead ecg showing extrasystoles",
     "an electrocardiogram with atrial or ventricular premature beats",
     "ecg with ectopic beats"],
    ["a twelve lead ecg showing ventricular hypertrophy",
     "an electrocardiogram indicating atrial or ventricular hypertrophy",
     "ecg with hypertrophy or chamber overload"],
    ["a twelve lead ecg showing myocardial ischemia",
     "an electrocardiogram indicating myocardial infarction or scar",
     "ecg with signs of ischemia or infarction"],
    ["a twelve lead ecg showing st and t wave changes",
     "an electrocardiogram with non-specific repolarization abnormality",
     "ecg with st-t changes"],
    ["a twelve lead ecg showing a paced rhythm",
     "an electrocardiogram with pacemaker spikes",
     "ecg with cardiac pacing"],
]

# ---- 3) ADD LABEL_ALIASES (read by data_fix._get_aliases) -----------------
# First token in each row == the lowercased 0/1 column name in labels.csv.
LABEL_ALIASES = [
    ["norm", "normal ecg", "normal"],
    ["afib_fl", "atrial fibrillation or flutter", "atrial fibrillation", "atrial flutter"],
    ["cd", "conduction disturbance", "conduction abnormality"],
    ["ectopy", "atrial or ventricular extrasystoles", "extrasystole", "pvc", "pac"],
    ["hyp", "cardiac hypertrophy", "hypertrophy", "overload"],
    ["mi", "myocardial ischemia or infarction", "ischemia", "myocardial infarction", "stemi"],
    ["sttc", "st t wave change", "st/t wave change", "repolarization"],
    ["pace", "cardiac pacing", "pacemaker"],
]
