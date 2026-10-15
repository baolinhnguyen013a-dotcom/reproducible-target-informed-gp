# WPT source scope used by the revised manuscript

Source: M. Asadi and A. M. Abazari, “A Data-Driven Model to Predict Mutual
Inductance Between Planar Coils With Arbitrary Specifications and Positions,”
*IET Electric Power Applications*, 2025, e70015,
doi:10.1049/elp2.70015.

The source PDF was read directly from `Desktop/elp2.70015.pdf`. Table 2 defines
14 inputs:

- 8 intrinsic geometry variables: transmitter and receiver polygon side count,
  turn count, spacing, and inner radius;
- 6 relative-pose variables: three translations and three rotations.

The source generated 15,000 numerical labels, retained 3,000 for testing, and
trained its regressors on 12,000. It studied mutual-inductance prediction, not
resistance-minimizing design.

The revised manuscript deliberately uses only the eight geometry variables.
The receiver pose is fixed at `(0, 0, 70 mm)` with zero rotation. There is no
misalignment optimization, tolerance envelope, or robustness claim. The paper's
`15,000 + 12` count is explicitly an *adapted workflow* in which the published
global data set is followed by 12 target-specific physical confirmations; it is
not attributed to the source authors.

The local oracle uses the source partial-inductance geometry and a copper
centerline resistance objective. In accordance with the source geometry text,
turn pitch is spacing plus wire width. This corrects the historical
`run_wpt_8d_geom_and_tolerance.py` script, which left the coil object's wire
width at zero. Historical 8-D/tolerance CSV files are retained for provenance
but are not evidence for the revised paper.
