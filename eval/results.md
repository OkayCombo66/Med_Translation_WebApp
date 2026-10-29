# Eval results — 2026-10-29

Model: `claude-sonnet-5-5` · reading level: 8th · documents: 19

## Summary

| Metric | Full pipeline | Naive prompt |
|---|---|---|
| Reading grade level (source: 7.5) | 5.9 | 6.0 |
| Numeric faithfulness | 100% | 98% |
| Unsupported claims about the patient | 1 of 638 checked | 66 of 500 checked |
| Abnormal results mentioned | 100% | 70% |
| Disclaimer present | 100% | 47% |
| Docs with MedlinePlus citations | 18/19 | — |

Full pipeline also flagged 12 general-background claims as not backed by its references. Average full-pipeline time: 13.6s per document.

## Per document

| File | Src grade | Full grade | Naive grade | Full numeric | Naive numeric | Full / naive unsupported patient claims | Abnormal covered (full / naive) |
|---|---|---|---|---|---|---|---|
| discharge_appendectomy.txt | 7.5 | 6.8 | 5.8 | 100% | 100% | 0 / 1 | 100% / 0% |
| discharge_copd_exacerbation.txt | 8.3 | 6.4 | 6.1 | 100% | 100% | 0 / 2 | n/a / n/a |
| discharge_diabetic_ketoacidosis.txt | 7.3 | 6.5 | 5.6 | 100% | 100% | 0 / 4 | 100% / 18% |
| discharge_heart_failure.txt | 6.8 | 6.3 | 5.4 | 100% | 93% | 0 / 3 | 100% / 100% |
| discharge_pneumonia.txt | 6.8 | 6.0 | 5.0 | 100% | 100% | 0 / 2 | n/a / n/a |
| lab_all_normal_physical.txt | 5.2 | 4.4 | 7.1 | 100% | 100% | 0 / 4 | n/a / n/a |
| lab_cbc_iron_deficiency.txt | 6.6 | 5.1 | 6.3 | 100% | 100% | 0 / 3 | 100% / 91% |
| lab_diabetes_a1c_followup.txt | 7.9 | 5.6 | 5.3 | 100% | 100% | 0 / 2 | 100% / 50% |
| lab_kidney_ckd_stage3.txt | 7.7 | 5.6 | 7.0 | 100% | 100% | 0 / 4 | 100% / 75% |
| lab_lipid_panel_borderline.txt | 8.0 | 6.4 | 6.6 | 100% | 95% | 0 / 5 | 100% / 75% |
| lab_liver_panel_elevated.txt | 10.2 | 6.6 | 6.2 | 100% | 100% | 0 / 4 | 100% / 100% |
| lab_thyroid_hypothyroid.txt | 7.3 | 6.3 | 4.8 | 100% | 100% | 0 / 5 | 100% / 100% |
| radiology_chest_xray_pneumonia.txt | 10.1 | 6.0 | 6.9 | 100% | 100% | 0 / 2 | n/a / n/a |
| radiology_ct_abdomen_kidney_stone.txt | 8.4 | 6.0 | 4.9 | 100% | 100% | 0 / 3 | n/a / n/a |
| radiology_mri_knee_meniscus_tear.txt | 10.3 | 7.6 | 6.2 | 100% | 100% | 1 / 5 | n/a / n/a |
| radiology_thyroid_ultrasound_nodule.txt | 5.4 | 5.2 | 5.0 | 100% | 75% | 0 / 2 | n/a / n/a |
| sample_lab_report.txt | 7.4 | 5.5 | 6.9 | 100% | 100% | 0 / 5 | 100% / 100% |
| tricky_mixed_narrative_and_labs.txt | 6.3 | 5.3 | 6.8 | 100% | 100% | 0 / 6 | 100% / 75% |
| tricky_ocr_noisy_bmp.txt | 5.1 | 5.1 | 6.8 | 100% | 91% | 0 / 4 | 100% / 50% |

## Numbers not found in the source

- `discharge_heart_failure.txt` — full: none; naive: 55, 70
- `lab_lipid_panel_borderline.txt` — full: none; naive: 30
- `radiology_thyroid_ultrasound_nodule.txt` — full: none; naive: 15, 30
- `tricky_ocr_noisy_bmp.txt` — full: none; naive: 911
