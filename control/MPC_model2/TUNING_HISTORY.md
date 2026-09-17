# Model 2 Tuning History

All Unity trials used the same `UnderwaterVision` scene, 20 Hz bridge, 0.12
m/s horizontal X PingPong fish trajectory, instantaneous endpoint reversal, and
the identified simulation dynamics:

```text
M_t = diag(26.07276, 26.79684, 26.07276)
D_L = diag(93.88006, 143.69195, 280.86849)
```

`tau_base` was zero and was not used by model 2 prediction, filtering, cost, or
normal tracking. The value is retained only as a compatibility field in the
shared dynamics type; model-2 safety fallback returns zero force.

## Short Screening

| Candidate | Position weights FRD | Velocity weights FRD | Terminal P/V | Constant-speed X MAE | Constant-speed 3D MAE | Saturation |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Old dual weights | `1600/700/15000` | `18/16/16` | `4/4` | `0.07801 m` | `0.08083 m` | `11.57%` |
| Position 5000 | `5000/1200/18000` | `18/20/20` | `4/4` | `0.05761 m` | `0.06093 m` | `16.53%` |
| Aggressive 12000 | `12000/3000/40000` | `2/4/4` | `8/1` | `0.04373 m` | `0.04582 m` | `17.77%` |
| Selected candidate | `60000/10000/120000` | `0.2/0.5/0.5` | `4/4` | `0.03349 m` | `0.03535 m` | `18.60%` |
| Same selected weights, `P/V=12/0.25` | `60000/10000/120000` | `0.2/0.5/0.5` | `12/0.25` | `0.03456 m` | `0.03641 m` | `18.60%` |

The independent terminal position/velocity multipliers were therefore kept as
an available experiment interface but not selected. The unified `4/4` setting
was slightly better under the same aggressive weights.

## Full Comparison

The selected model-2 candidate was rerun for 65 seconds and two reversals:

| Metric | Model 2 selected | Previous dual-model indirect baseline |
| --- | ---: | ---: |
| Constant-speed X MAE | `0.03370 m` | `0.00203 m` |
| Constant-speed X p95 | `0.03502 m` | `0.00644 m` |
| Constant-speed 3D MAE | `0.03482 m` | `0.00269 m` |
| Full-run 3D MAE | `0.03991 m` | not recorded in the same JSON |
| Error samples <= `0.03 m` | `0.77%` | `86.10%` |
| QP success | `100%` | `100%` |
| Force saturation | `5.15%` | `9.98%` |

The model-2 result is materially different from the earlier dual-model result,
so a horizontal S-trajectory follow-up was not used to manufacture a visual
difference. The remaining steady offset is expected from the requested pure
model-2 structure: a continuous fish-motion force cannot be represented as a
matched baseline. Adding `tau_base` back into model 2 would improve the number
but would no longer be the requested mathematics.

Artifacts are stored in
`FinsSim/ros2_ws/data/mpc_model2_tuning_20260807/`.
