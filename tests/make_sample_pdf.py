"""Generate a small fake paper so the test suite needs no downloads.

It deliberately reproduces the things that break naive PDF RAG: hard line
wraps mid-sentence, words hyphenated across a line break, numbered section
headings, a fact that straddles a page boundary, and a long bibliography.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

DEST = Path(__file__).resolve().parent / "sample_paper.pdf"

BODY = """Abstract
We present GLIDE, a lightweight method for detecting anomalous readings in
implanted orthopedic sensor telemetry. GLIDE combines a per-patient base-
line with a shared population prior, and reaches an F1 of 0.87 on our held-
out cohort, compared with 0.71 for the strongest baseline.

1 Introduction
Continuous telemetry from instrumented implants produces long, irregularly
sampled series that are dominated by patient-specific variation. Existing
anomaly detectors treat each patient independently, which discards the popu-
lation signal, or pool all patients, which drowns the individual signal. We
argue that both are needed, and that the trade-off can be made explicit.

2 Related Work
Prior work on physiological anomaly detection falls into two families. Recon-
struction methods train an autoencoder per subject. Forecasting methods fit
a predictive model and flag large residuals. Neither family reports results on
implanted device telemetry, where the sampling rate is two orders of magni-
tude lower than in wearable settings.

3 Method
GLIDE fits a hierarchical model in which each patient's baseline is drawn
from a population prior. The anomaly score for a reading is its negative log
likelihood under the posterior predictive distribution. We set the shrinkage
parameter lambda to 0.4, chosen on the validation split.

3.1 Population Prior
The prior is estimated by pooling all training patients and fitting a Student-t
distribution, which we found more robust to outliers than a Gaussian.

3.2 Per-Patient Baseline
Each patient's baseline is updated online with an exponential forgetting fac-
tor of 0.98, so that a genuine change in a patient's physiology is absorbed
within roughly fifty readings rather than being flagged indefinitely.

4 Data
We use the SENTINEL cohort, which contains 1,284 patients and 4.6 million
individual readings collected over 26 months across eleven surgical centres.
Readings arrive roughly every four hours. Labels were assigned by two ortho-
pedic surgeons, with disagreements resolved by a third reviewer; inter-rater
agreement was 0.81 by Cohen's kappa.

5 Experiments
We compare GLIDE against three baselines: a per-patient autoencoder, a
pooled gradient boosted tree, and a seasonal ARIMA residual detector. All
models are trained on the same split, with 70 percent of patients used for
training, 10 percent for validation and 20 percent held out for testing. The
split is by patient rather than by reading, so no patient appears in more than
one partition.

5.1 Main Results
GLIDE reaches an F1 of 0.87, against 0.71 for the pooled gradient boosted
tree, 0.66 for the per-patient autoencoder and 0.58 for the ARIMA residual
detector. Precision is 0.84 and recall is 0.90.

5.2 Ablations
Removing the population prior drops F1 to 0.74. Removing the per-patient
baseline drops it to 0.69. The two components are therefore complemen-
tary, and neither alone accounts for the improvement over the baselines.

6 Discussion
The gain comes mostly from recall on patients with fewer than two hundred
readings, where a per-patient model has too little history to calibrate. On
patients with long histories the three methods are within two points of each
other.

7 Limitations
The SENTINEL cohort is drawn from a single device family, and all eleven
centres are in one country, so the population prior may not transfer. We did
not evaluate on paediatric patients. Labels reflect surgeon judgement rather
than a confirmed clinical outcome.

8 Conclusion
We showed that combining a population prior with a per-patient baseline
improves anomaly detection on implanted sensor telemetry, and that both
components are necessary.

References
[1] A. Almeida and B. Chen. Autoencoders for physiological anomaly detec-
tion. Journal of Biomedical Signal Processing, 2021.
[2] C. Duarte, E. Fitzgerald and G. Haas. Pooled models for wearable time
series. Proceedings of the Conference on Health Informatics, 2022.
[3] H. Iyer and J. Kowalski. Seasonal ARIMA for residual anomaly scoring.
Transactions on Biomedical Engineering, 2020.
[4] K. Lindqvist. Robust priors with heavy tailed distributions. Statistics and
Computing, 2019.
[5] M. Nakamura, O. Pereira and P. Quinn. Implanted device telemetry at
scale. Nature Digital Medicine, 2023.
[6] R. Svensson. Hierarchical shrinkage for small-sample subjects. Journal of
Machine Learning Research, 2018.
[7] T. Ueda and U. Vogel. Inter-rater agreement in surgical labelling. Medical
Decision Making, 2021.
"""


def build(dest: Path = DEST) -> Path:
    pdf = canvas.Canvas(str(dest), pagesize=LETTER)
    width, height = LETTER
    margin = inch
    leading = 14
    y = height - margin

    for line in BODY.split("\n"):
        if y < margin:
            pdf.showPage()
            y = height - margin
        stripped = line.strip()
        is_heading = stripped and (
            stripped[0].isdigit() or stripped in {"Abstract", "References"}
        ) and len(stripped) < 40
        pdf.setFont("Helvetica-Bold" if is_heading else "Helvetica", 11)
        pdf.drawString(margin, y, line)
        y -= leading

    pdf.save()
    return dest


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
