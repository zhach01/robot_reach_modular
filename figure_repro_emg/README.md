# §11 EMG figure generators (model_based paper)

Version-controlled copies of the scripts that generate the four EMG figures in
`overleaf_model_based/sections/11_validation_modelbased.tex`.

| Figure | Script |
|---|---|
| emg_diagnostic.png, emg_meanprofile.png | emg_compact_figs.py |
| emg_improved_overlay.png | emg_improved_overlay_fig.py |
| emg_crosssubject.pdf | emg_xsub_worker.py (×10, parallel) + emg_xsub_plot.py |

**Dependency:** these require the `robot_reach_modular_EMG_cocon` library variant,
which adds the deceleration-gated co-contraction `PDIFParams` fields
(`cocon_level / cocon_vel_gain / cocon_vel_ref / cocon_decel_only`) to
`controller/numpy/pd_if_controller.py`. They are NOT runnable against this repo's
stock controller. Data: `kinarm_dataset/raw_lucchetti/HS01..HS10.mat` (Lucchetti
2025, doi:10.1038/s41597-025-06174-3).

Reproduced values (match the §11 text): cross-subject model–EMG correlation
0.34 → 0.45; shoulder extensor 0.21 → 0.44; triceps 0.16 → 0.31; 8/10 subjects
improve to positive r.
