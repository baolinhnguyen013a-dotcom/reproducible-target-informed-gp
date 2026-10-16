# Frozen S3 weight selection

The development experiment used seeds 31261011--31261020, all integer targets
from 1 to 87 mH, total budget B=20 (6 initial observations, 2 acquisition
observations, and 12 GP-ranked validations), and ten-member neural ensembles.

The primary selection criterion was all-case mean bounded loss.  The observed
values were:

| Acquisition | Mean loss | Qualification | Exact optimum | Top ten |
|---|---:|---:|---:|---:|
| Pure GP-cEI | 0.120446 | 91.494% | 51.379% | 90.460% |
| cEI + NTK, beta=0.05 | 0.099161 | 93.678% | 53.333% | 92.644% |
| cEI + NTK, beta=0.10 | 0.100229 | 93.563% | 53.103% | 92.529% |
| cEI + NTK, beta=0.20 | **0.099007** | **93.678%** | 53.218% | **92.644%** |

Accordingly, **beta=0.20 is frozen before confirmation**.  Its margin over
beta=0.05 is only 0.000154 in mean loss and does not establish that 0.20 is a
generally optimal weight.  Against pure cEI, its ten-seed paired mean-loss
difference is -0.021439; the seed-bootstrap 95% interval
[-0.057649, 0.001521] includes zero.  Fresh-seed confirmation is therefore
required.

No confirmation result may be used to revise beta.  All confirmation methods
must use identical initial observations, total budgets, target set, GP-only
validation ranking, and physical-oracle decision rule.
