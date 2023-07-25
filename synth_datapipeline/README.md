### what this contains
- raw synthetic data (robot only) -> nerfstudio format
- nerfstudio format (robot only) -> enerf
- nerfstudio format (any nerfstudio) -> enerf

### How to use
#### For robot only  (raw robot synthetic -> nerfstudio) (run in order)
- run format_ecam_set.py to format event camera dataset
- run format_colcam_set.py to format rgb camera dataset

#### For robot only (nerfstudio -> enerf)
- run to_enerf.py

#### For any nerfstudio (nerfstudio -> enerf):
- run to_enerf_general.py
