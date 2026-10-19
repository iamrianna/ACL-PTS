# PTS Workstation

Streamlit PACS-style workstation for **fully automated** posterior tibial slope (PTS) on sagittal knee MRI
(e.g. 1.5T PD-weighted, slice 16/27). Made by Rianna-Maria Tanase.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs
`.pck/.pkl` (numpy arrays only, restricted unpickler), `.npy`, `.jpg`, `.png`. 2D slice or 3D volume.
Set the "anterior side" of the image in the sidebar (metadata, not a measurement).

## What it does automatically
1. Tibia segmentation (classical baseline; swap in a trained model via `TIBIA_SEG_ONNX=path.onnx`).
2. Shaft axis + plateau line -> PTS (positive = plateau descends posteriorly).
3. 95% CI from an ensemble (threshold perturbation + window jitter + bootstrap), auto-QC flag PASS/REVIEW.
4. Risk tier Low/High (default threshold 12 deg, editable; flags borderline if CI crosses it).
5. Biomechanics: anterior shear = F*sin(PTS); ACL load increase vs a reference slope (simplified static model,
   all parameters editable; not validated).
6. ACL rupture status: needs a trained classifier (`models/acl_classifier.onnx`, input 1x1x256x256 in [0,1],
   output tear probability). Without it the app says "NOT ASSESSED" and shows risk only.
7. Validation tab: upload cases + CSV (`filename,reference_pts[,reference_acl]`) -> MAE, RMSE, bias, limits of
   agreement, ICC(2,1), Pearson r, % within 2/3 deg, CI coverage, tier sensitivity/specificity, Bland-Altman.
8. Exports: watermarked PNG, CSV, PDF.

## Limitations
Research prototype, not a medical device. Assumes isotropic pixels. The classical segmentation is a baseline and
must be validated on your own data before any conclusions.
