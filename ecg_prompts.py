"""
LLM-style prompt ensembles for BiomedCoOp on PTB-XL.

BiomedCoOp's two extra losses (SCCM + KDSP) need, for every class, a set of
descriptive sentences that a strong LLM (GPT-4 in the paper) would write about
how that class looks. In the official repo these live in
`trainers/prompt_templates.py` as `BIOMEDCOOP_TEMPLATES[classname] = [ ... ]`.

Here we provide the same structure for the 5 PTB-XL diagnostic superclasses.
The text is clinically-grounded description of the 12-lead ECG *image* (that is
what BiomedCLIP sees after `ecg_to_image.py` renders the waveform). You can
regenerate/extend these with an LLM of your choice; more, more-diverse prompts
generally help, and KDSP's outlier pruning will drop the unhelpful ones.

Keys are the PTB-XL superclass codes used everywhere else in the repo
(config.CLASSES). `readable_name(code)` gives the short phrase that goes into
the *learnable* prompt ("a photo of a <name>.").
"""

# Short class phrase inserted after the learnable context tokens.
# (kept separate from config.CLASS_DESCRIPTIONS so you can tune prompt wording
#  without touching the zero-shot captions.)
READABLE_NAMES = {
    "NORM": "normal ECG",
    "MI":   "myocardial infarction",
    "STTC": "ST/T wave change",
    "CD":   "conduction disturbance",
    "HYP":  "cardiac hypertrophy",
}


def readable_name(code):
    return READABLE_NAMES.get(code, code)


BIOMEDCOOP_TEMPLATES = {
    "NORM": [
        "A normal 12-lead ECG with regular sinus rhythm and a consistent rate between 60 and 100 beats per minute.",
        "This 12-lead electrocardiogram shows normal sinus rhythm with upright P waves preceding every QRS complex.",
        "A normal ECG tracing with narrow QRS complexes, normal axis, and no ST-segment deviation.",
        "The ECG demonstrates a normal PR interval, normal QRS duration, and a normal QT interval.",
        "A healthy 12-lead ECG with regularly spaced R waves and stable baseline across all leads.",
        "Normal electrocardiogram showing symmetric, upright T waves and no pathological Q waves.",
        "A twelve-lead ECG within normal limits, with normal R-wave progression across the precordial leads.",
        "This ECG shows a normal cardiac rhythm with no signs of ischemia, hypertrophy, or conduction block.",
        "A normal sinus rhythm tracing with each P wave followed by a QRS complex at a constant interval.",
        "The 12-lead ECG appears entirely normal, with clean waveforms and no arrhythmia.",
        "A normal ECG with an isoelectric ST segment and physiologic T-wave morphology in all leads.",
        "Standard 12-lead electrocardiogram of a healthy heart with normal intervals and normal voltages.",
        "The tracing shows regular rhythm, normal atrial and ventricular activation, and no abnormalities.",
        "A normal ECG with well-defined P-QRS-T complexes and no ectopic beats.",
        "This electrocardiogram is unremarkable, showing normal conduction from atria to ventricles.",
        "A normal 12-lead ECG with consistent RR intervals and a stable, flat baseline.",
        "The ECG shows physiologic waveform amplitudes and no ST elevation or depression.",
        "A clean electrocardiogram with normal sinus rhythm and no morphological abnormalities.",
        "Twelve-lead ECG demonstrating a healthy conduction system and normal repolarization.",
        "A normal tracing with narrow QRS, upright P waves in lead II, and normal precordial progression.",
        "The electrocardiogram reflects a structurally and electrically normal heart.",
        "A normal ECG image with evenly spaced complexes and no evidence of infarction or block.",
        "This 12-lead recording shows regular, normal-rate sinus rhythm with clean morphology.",
        "A typical normal ECG with no Q waves, no ST changes, and no T-wave inversion.",
        "Normal cardiac electrical activity captured on a standard 12-lead electrocardiogram.",
        "A normal ECG with balanced limb-lead voltages and normal precordial R-wave amplitude.",
        "The tracing is a normal 12-lead ECG with no arrhythmic or ischemic features.",
        "A healthy electrocardiogram showing normal atrioventricular conduction and repolarization.",
        "A normal-appearing 12-lead ECG with regular rhythm and no waveform distortion.",
        "This ECG is within normal limits across rate, rhythm, axis, intervals, and morphology.",
    ],
    "MI": [
        "A 12-lead ECG showing myocardial infarction with pathological Q waves in the affected leads.",
        "This electrocardiogram demonstrates acute myocardial infarction with ST-segment elevation.",
        "An ECG of myocardial infarction with deep Q waves and loss of R-wave progression.",
        "The tracing shows signs of infarction, including ST elevation and reciprocal ST depression.",
        "A 12-lead ECG consistent with an old myocardial infarction, showing persistent Q waves.",
        "Electrocardiogram of anterior myocardial infarction with ST elevation in the precordial leads.",
        "An inferior myocardial infarction pattern with ST-segment elevation in leads II, III, and aVF.",
        "This ECG shows myocardial infarction with abnormal Q waves and T-wave inversion.",
        "A tracing of acute MI with hyperacute T waves evolving into ST-segment elevation.",
        "The electrocardiogram reveals infarction with poor R-wave progression across the chest leads.",
        "A 12-lead ECG showing a healed infarct with residual Q waves and flattened T waves.",
        "Myocardial infarction on ECG with convex ST elevation and reciprocal changes.",
        "This ECG demonstrates a lateral infarction with Q waves in leads I and aVL.",
        "An electrocardiogram of ischemic injury with ST elevation and developing Q waves.",
        "The tracing shows a septal myocardial infarction with loss of anteroseptal R waves.",
        "A 12-lead ECG with QS complexes indicating transmural myocardial infarction.",
        "Electrocardiographic evidence of infarction with pathological Q waves and ST-T abnormalities.",
        "This ECG shows an evolving myocardial infarction with ST elevation and T-wave inversion.",
        "A tracing of prior infarction with persistent Q waves and no acute ST changes.",
        "Myocardial infarction pattern on a 12-lead ECG with abnormal repolarization.",
        "The ECG reveals a posterior infarction with tall R waves and ST depression in V1-V3.",
        "A 12-lead electrocardiogram showing acute coronary occlusion with marked ST elevation.",
        "This ECG demonstrates infarct-related Q waves greater than 40 milliseconds in duration.",
        "An anterolateral myocardial infarction with ST elevation across multiple precordial leads.",
        "The tracing shows myocardial necrosis with fixed Q waves and diminished R-wave amplitude.",
        "A 12-lead ECG of infarction with coved ST-segment elevation and hyperacute T waves.",
        "Electrocardiogram consistent with myocardial infarction and secondary ST-T changes.",
        "This ECG shows regional ST elevation with reciprocal depression indicating acute infarction.",
        "A myocardial infarction tracing with abnormal Q waves and inverted, asymmetric T waves.",
        "The 12-lead ECG demonstrates an ischemic infarct pattern with ST-segment shifts.",
    ],
    "STTC": [
        "A 12-lead ECG showing ST-segment and T-wave changes suggesting myocardial ischemia.",
        "This electrocardiogram demonstrates ST-segment depression across several leads.",
        "An ECG with widespread T-wave inversion consistent with repolarization abnormality.",
        "The tracing shows nonspecific ST-T changes with flattened T waves.",
        "A 12-lead ECG with horizontal ST-segment depression indicating ischemia.",
        "Electrocardiogram showing symmetric, deeply inverted T waves in the precordial leads.",
        "This ECG demonstrates ST-segment and T-wave abnormalities without pathological Q waves.",
        "A tracing with downsloping ST depression and biphasic T waves.",
        "The electrocardiogram reveals diffuse ST-T changes suggestive of subendocardial ischemia.",
        "A 12-lead ECG with T-wave flattening and minor ST-segment deviation.",
        "ST/T wave changes on ECG with inverted T waves in the lateral leads.",
        "This ECG shows repolarization abnormalities with sagging ST segments.",
        "A tracing of ischemic ST-T changes with depressed J points.",
        "The electrocardiogram demonstrates nonspecific T-wave abnormalities across the tracing.",
        "A 12-lead ECG with ST depression and T-wave inversion in the inferior leads.",
        "Electrocardiographic ST-segment changes consistent with myocardial strain.",
        "This ECG reveals prominent T-wave inversions without evidence of infarction.",
        "A tracing with subtle ST-segment elevation and abnormal T-wave morphology.",
        "The ECG shows ST-T wave changes indicating altered ventricular repolarization.",
        "A 12-lead electrocardiogram with flattened and inverted T waves in multiple leads.",
        "ST and T-wave abnormalities on ECG suggestive of ischemia or electrolyte disturbance.",
        "This ECG demonstrates lateral ST depression with asymmetric T-wave inversion.",
        "A tracing showing nonspecific ST-segment shifts and low-amplitude T waves.",
        "The electrocardiogram reveals diffuse repolarization changes with ST-T abnormalities.",
        "A 12-lead ECG with ischemic-appearing ST depression during the recording.",
        "Electrocardiogram with T-wave inversions and mild ST-segment depression.",
        "This ECG shows widespread ST-T changes without acute infarction.",
        "A tracing of repolarization disturbance with abnormal ST segments and T waves.",
        "The 12-lead ECG demonstrates ST-segment depression and flattened T waves.",
        "An electrocardiogram with nonspecific ST-T wave changes across the precordium.",
    ],
    "CD": [
        "A 12-lead ECG showing a conduction disturbance with a widened QRS complex.",
        "This electrocardiogram demonstrates a bundle branch block with prolonged QRS duration.",
        "An ECG of right bundle branch block with an rSR' pattern in lead V1.",
        "The tracing shows left bundle branch block with broad, notched QRS complexes.",
        "A 12-lead ECG with first-degree AV block and a prolonged PR interval.",
        "Electrocardiogram showing a conduction abnormality with delayed ventricular activation.",
        "This ECG demonstrates an intraventricular conduction delay with a wide QRS.",
        "A tracing of atrioventricular block with dissociated P waves and QRS complexes.",
        "The electrocardiogram reveals a left anterior fascicular block with left axis deviation.",
        "A 12-lead ECG with a bundle branch block pattern and secondary ST-T changes.",
        "Conduction disturbance on ECG with a QRS duration exceeding 120 milliseconds.",
        "This ECG shows second-degree AV block with intermittently dropped QRS complexes.",
        "A tracing of complete heart block with independent atrial and ventricular rhythms.",
        "The electrocardiogram demonstrates a right bundle branch block with a wide terminal S wave.",
        "A 12-lead ECG with left bundle branch block and a dominant S wave in V1.",
        "Electrocardiographic evidence of a conduction delay with abnormal QRS morphology.",
        "This ECG reveals a fascicular block with a characteristic frontal-plane axis shift.",
        "A tracing showing prolonged AV conduction with a lengthened PR segment.",
        "The ECG demonstrates a nonspecific intraventricular conduction disturbance.",
        "A 12-lead electrocardiogram with a wide, bizarre QRS from a conduction block.",
        "Bundle branch block on ECG with discordant ST-segment and T-wave changes.",
        "This ECG shows a bifascicular block combining right bundle branch and fascicular block.",
        "A tracing of AV nodal block with a slow ventricular response.",
        "The electrocardiogram reveals delayed conduction with a broadened QRS complex.",
        "A 12-lead ECG with an incomplete right bundle branch block pattern.",
        "Electrocardiogram demonstrating a conduction disturbance and abnormal QRS width.",
        "This ECG shows Wenckebach periodicity with progressive PR prolongation.",
        "A tracing of a wide-QRS rhythm caused by an intraventricular conduction block.",
        "The 12-lead ECG demonstrates a bundle branch block with prolonged QRS and ST-T changes.",
        "An electrocardiogram with an atrioventricular conduction abnormality and widened complexes.",
    ],
    "HYP": [
        "A 12-lead ECG showing left ventricular hypertrophy with increased QRS voltage.",
        "This electrocardiogram demonstrates high-amplitude R waves consistent with hypertrophy.",
        "An ECG of left ventricular hypertrophy with a strain pattern of ST depression and T inversion.",
        "The tracing shows tall R waves in the lateral leads meeting voltage criteria for hypertrophy.",
        "A 12-lead ECG with right ventricular hypertrophy and right axis deviation.",
        "Electrocardiogram showing deep S waves in V1-V2 and tall R waves in V5-V6.",
        "This ECG demonstrates increased ventricular voltage indicating chamber enlargement.",
        "A tracing of left atrial enlargement with a broad, notched P wave.",
        "The electrocardiogram reveals hypertrophy with prominent precordial QRS amplitude.",
        "A 12-lead ECG with voltage criteria for left ventricular hypertrophy and a strain pattern.",
        "Cardiac hypertrophy on ECG with tall R waves and secondary repolarization changes.",
        "This ECG shows right ventricular hypertrophy with a dominant R wave in V1.",
        "A tracing of biventricular hypertrophy with high voltages in both directions.",
        "The electrocardiogram demonstrates left ventricular hypertrophy by Sokolow-Lyon voltage.",
        "A 12-lead ECG with increased QRS amplitude and downsloping ST segments from hypertrophy.",
        "Electrocardiographic evidence of ventricular enlargement with elevated voltages.",
        "This ECG reveals hypertrophy with tall R waves and inverted T waves in lateral leads.",
        "A tracing showing left atrial abnormality and left ventricular hypertrophy.",
        "The ECG demonstrates high-amplitude complexes consistent with myocardial hypertrophy.",
        "A 12-lead electrocardiogram with a strain pattern accompanying ventricular hypertrophy.",
        "Ventricular hypertrophy on ECG with prominent voltages and repolarization abnormality.",
        "This ECG shows right atrial enlargement with tall, peaked P waves.",
        "A tracing of left ventricular hypertrophy with deep S waves and tall R waves.",
        "The electrocardiogram reveals hypertrophic voltage criteria with ST-T strain changes.",
        "A 12-lead ECG with increased R-wave amplitude indicating chamber hypertrophy.",
        "Electrocardiogram demonstrating ventricular hypertrophy and secondary ST depression.",
        "This ECG shows enlarged ventricular voltages with a left-sided strain pattern.",
        "A tracing of cardiac hypertrophy with tall precordial R waves and T-wave inversion.",
        "The 12-lead ECG demonstrates hypertrophy with high QRS voltage and abnormal repolarization.",
        "An electrocardiogram with voltage and repolarization findings typical of hypertrophy.",
    ],
}


def get_templates(classes, n_prompts):
    """Return an (n_classes) list of prompt-lists, each truncated to n_prompts.

    Raises if a class has fewer than n_prompts templates so failures are loud.
    """
    out = []
    for c in classes:
        if c not in BIOMEDCOOP_TEMPLATES:
            raise KeyError(f"No BIOMEDCOOP_TEMPLATES for class '{c}'")
        prompts = BIOMEDCOOP_TEMPLATES[c]
        if len(prompts) < n_prompts:
            raise ValueError(
                f"class '{c}' has {len(prompts)} templates but n_prompts={n_prompts}. "
                f"Add more prompts or lower N_PROMPTS."
            )
        out.append(prompts[:n_prompts])
    return out
