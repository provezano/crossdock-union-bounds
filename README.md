# Code and data for: Union-of-predecessors bounds and a forward–backward local search for multi-door cross-dock truck scheduling

**Author:** Felipe Provezano Coutinho (Independent researcher), ORCID [0000-0001-7206-5856](https://orcid.org/0000-0001-7206-5856)

> **This repository holds the code only.** The full archive (per-instance bounds and best schedules for all 10,500 benchmark instances, generated instance sets, per-run outputs, trained networks and the SHA-256 manifest) is on Zenodo: [10.5281/zenodo.23933868](https://doi.org/10.5281/zenodo.23933868). Paths under `data/` and the per-run output folders mentioned below refer to that archive.


This package lets you reproduce the computational results of the paper (P | M_j, prec | Cmax on the benchmark of
Cota et al. 2016). It contains:
- the code of the bounds, the upper-bound method (FB-ILS), the exact method and the re-implemented rivals;
- per-instance bounds and best schedules for all 10,500 benchmark instances;
- the generated instance sets of the confirmation experiments;
- the scripts that build the paper's tables.

## Layout
| Path | Contents |
|---|---|
| `code/lower_bounds/` | The union-of-predecessors bound (`g1.py`), certificates, census driver and its output `analysis_full_10500.txt`. |
| `code/decoder/` | Schedule decoder and checker (`cdc.py`), constructive orders and the first local search (`g2.py`). |
| `code/eacdc_matched_tim/` | EACDC harness, the time-indexed LP (`tim_lp.py`), tuning and cost-per-evaluation runs. |
| `code/confirmation_eacdc/` | Fresh-instance generator and the matched comparison with EACDC. |
| `code/exact_cpsat/` | Exact methods (CP-SAT), energetic reasoning, two-door set, CP-SAT as an upper bound. |
| `code/fbils_census_exact_cpuc/` | FB-ILS census, its confirmation, structured instances, and the exact method CP-UC (`exact_cpuc_main.py`, `g6cp.py`). |
| `code/fbils_frozen/` | Frozen FB-ILS / iterated greedy: C source `c/srch.c` and its Python wrapper `srch.py`. |
| `code/hgr_comparison/`, `code/bound_analyses/hgr/` | Our re-implementation of HgR (Fonseca, Nogueira & Ravetti 2019) and its comparison runs. |
| `code/eacdcml_rebuild/` | Our rebuild of EACDCml (Nogueira et al. 2024): calibration, network and comparison runs; trained networks `model*.pt`. |
| `code/eacdc_public/` | Wrapper code for the public EACDC `GA.py` (not included, see below). |
| `code/bound_analyses/{det,lp,dom,cx}` | Deterministic bound census, LP bound, dominance checks, operation counts. |
| `code/window_bound/`, `code/margin_analysis/` | The window-inference bound UPB-W, and the margin/headroom analysis. |
| `code/exact_fresh/exact/` | CP-UC against CP-LB, CP-plain and the time-indexed MIP on 300 fresh instances: driver, MIP model, analysis, per-run outputs (`out_fresh/`) and the analysis output `ana_fresh.txt`. |
| `code/tim_lp_strong/` | Time-indexed LP with strengthened makespan linking (`tim_lp_strong.py`), its driver, analysis, per-instance outputs (`out_small/`, `out_real/`, `out_large/`) and the analysis output `ana_final.txt`. |
| `code/anytime_ig/` | Anytime profile at 1×/10×/100× evaluations and the tuned iterated greedy, with its outputs (`analysis.json`). |
| `code/paper_tables/` | Scripts that write the generated LaTeX tables. |
| `data/` | CSV files and generated instances (see "Data dictionary"). |
| `env/` | Package versions. |

Analysis outputs (`ana_*.txt`, `analysis*.json`) are shipped next to the scripts that produced them, so the table
scripts run without re-running the experiments.

## Requirements
- CPython 3.12 with `env/requirements_main_py312.txt` (numpy, scipy, OR-Tools 9.15, DEAP 1.4.4, PyTorch 2.15, Optuna 5.0, highspy).
- CPython 3.14 with `env/requirements_hgr_py314.txt`, for the HgR re-implementation.
- A C compiler (clang or gcc).
- All experiments ran with one thread per process (`OMP_NUM_THREADS=1`; CP-SAT and HiGHS with one worker).

## Third-party inputs (not included)
- **Benchmark instances** (Cota et al. 2016): `Data_Set.rar` in https://github.com/yotitinogs/EA-ML, or the repository of
  Fonseca et al. (2019). Unpack to `code/benchmark_data/data/Dados/<group>/<n1>jobs/<seed>.dat`. The group folders are
  `2maquinas`, `4maquinas`, `10maquinas`, `[2-4]maquinas` and `[2-10]`.
- **EACDC public code**: `GA.py` from https://github.com/yotitinogs/EA-ML, commit
  `62140295b9a07cb1123867b05d0db92b1aef837f` (sha256 `e280153e58a45d20c44784f3395698d8d35ff33785dd79d752297c07e614cdf6`).
  Put it at `code/eacdc_public/third/EA-ML/GA.py`. It has no licence, so it is not redistributed. Scripts that
  run it must be started from that directory with `python -I`.

## Set-up (from a clean copy)
```
python3.12 -m venv venv && venv/bin/pip install -r env/requirements_main_py312.txt
python3.14 -m venv venv_hgr && venv_hgr/bin/pip install -r env/requirements_hgr_py314.txt
# compiled search library (macOS; on Linux build code/fbils_census_exact_cpuc/c/srch.c as libsrch.so and use code/fbils_census_exact_cpuc/srch.py)
clang -O3 -mcpu=native -shared -fPIC code/fbils_frozen/c/srch.c -o code/fbils_frozen/c/libsrch.dylib
# generated instances at the paths the drivers expect
D=data/instances_generated
ln -s ../../$D/confirm_seed20261103 code/confirmation_eacdc/data_new
ln -s ../../$D/fbils_confirm_seed20261105   code/fbils_census_exact_cpuc/data_fbils_confirm
ln -s ../../$D/structured_seed20261104 code/fbils_census_exact_cpuc/data_struct
ln -s ../../$D/hgr_seed20261107     code/hgr_comparison/data
ln -s ../../../$D/exact_fresh_seed20261193 code/exact_fresh/exact/data_fresh
```

## Reproducing the tables
**Generated tables.** Write the 8 generated tables of the paper into `tables_out/`:
```
python3 -I code/paper_tables/make_tables.py tables_out
python3 -I code/paper_tables/make_tab_lb_census_det.py tables_out
```
The output is identical to the paper's tables (checked from a clean copy). The other tables are hand-built from the outputs listed below.

| Paper table | Produced by | Output read |
|---|---|---|
| LB census (`tab:lbcensus`) | `lower_bounds/run_lb.py`, `ana.py`; deterministic version `bound_analyses/det/run_det_all.py` | `lower_bounds/analysis_full_10500.txt`; `data/census_upb_det_10500.csv` |
| Earlier bounds (`tab:lbbaselines`) | `confirmation_eacdc/g1_nonown.py` | `g1_nonown.txt` |
| Energetic reasoning (`tab:er`), two-door set (`tab:twodock`), CP-SAT as UB (`tab:cpsatub`) | `exact_cpsat/run_er.py`, `g2dock.py`, `g5_ub.py`, `ana_g5.py` | `ana_er.txt`, `ana_dock.txt`, `ana_ub.txt` |
| Exact gaps and arms (`tab:exactgaps`, `tab:exactarms`) | `exact_cpsat/exact_main.py`, `ana_g5.py` | `ana_exact_all.txt` |
| CP-UC on the exact set | `fbils_census_exact_cpuc/exact_cpuc_main.py`, `ana_exact_cpuc.py` | `ana_exact_cpuc.txt` |
| ILS vs EACDC matched (`tab:ubmatched`) | `confirmation_eacdc/gen_new.py`, `g4_match.py`, `ana_match.py` | `ana_confirm.txt`, `ana_m105.txt` |
| FB-ILS confirmation (`tab:endpoints` B) | `fbils_census_exact_cpuc/gen_fbils_confirm.py`, `fbils_frozen/confirm_fbils.py`, `fbils_census_exact_cpuc/ana_conf_fbils.py` | `ana_conf_fbils.txt` |
| EACDCml comparison (`tab:endpoints` C, D) | `eacdcml_rebuild/match.py`, `ana_match.py`, `sens10x.py` | `out_match_analysis.json`, `sens10x_analysis.json` |
| HgR comparison (`tab:hgr`) | `hgr_comparison/gen_fresh.py`, `run_conf.py`, `ana_conf.py` | `analysis.txt` |
| Structured instances | `fbils_census_exact_cpuc/gen_struct.py`, `struct_run.py`, `match_struct.py`, `ana_struct.py` | `ana_struct.txt` |
| FB-ILS census, new best-known UBs | `fbils_census_exact_cpuc/fbils_census.py OUT SHARD NSH 1 10`, `ana_ils.py` | `ana_census_fbils.txt`; `data/census_*.csv` |
| Anytime profile and tuned IG (`tab:anytime`) | `anytime_ig/ig_tune.py`, `ig_select.py`, `hgr_any.py`, `anytime.py`, `igt_run.py`, `fill_first_schedule.py`, `ana_anytime.py` | `analysis.json` |
| UPB-W (`tab:wib`) | `window_bound/wib.py`, `wib_run.py`, `ana_wib.py` | — (re-run) |
| Margin–headroom (`tab:margin`) | `margin_analysis/ana_margin.py`, `ana_final.py` | `ana_margin.txt`, `ana_final.txt` |
| Cost per evaluation (`tab:complexity`) | `eacdc_matched_tim/complexity.py`; `bound_analyses/cx/opcount.py` | `complexity_local.jsonl`, `opcount.txt` |
| TIM-LP (`tab:tim`), tuning (`tab:tuning`) | `eacdc_matched_tim/run_tim.py`, `ana_tim.py`; `tune.py`, `ana_tune.py` | — (re-run) |
| CP-UC on 300 fresh instances | `exact_fresh/exact/exact_fresh.py gen`, `exact_fresh.py run KEY ARMS OUTDIR 300`, `ana_fresh.py out_fresh 300` | `ana_fresh.txt`; `data/exact_fresh300_per_instance.csv` |
| Strengthened TIM-LP | `tim_lp_strong/run_strong.py {small,real,large} …`, `ana_strong.py out_small out_real out_large` | `ana_final.txt`; `data/tim_lp_strengthened_per_instance.csv` |

Run outputs are written next to the scripts (`out_*`). Instance lists (`list_*.json`, `sample105.json`) are relative to their script directory.

## Data dictionary
**`data/census_bounds_10500.csv`** (one row per benchmark instance)
- `instance`: `<group>_<n1>_<file seed>`. `dock_group` ∈ {m2, m4, m10, u24, u210}; `n1`, `n2`; `m1`, `m2` (inbound and outbound doors).
- `lb_chen_song`, `lb_pch`: earlier bounds. `lb_star`: our bound LB* = max(UPB(I), UPB(R(I))). `lb_proven`: the best proven LB (LB* or an exact run).
- `ub_hgr_published`: the HgR UB of Fonseca et al. (2019). `ub_fbils_10s`: FB-ILS, 10 CPU s, seed 1.
- `ub_best_known` and `best_known_source`.
- `proven_optimal`: 1 if an exact run proved optimality (165 instances). Optimality is certified on 188 instances, those with `lb_proven` = `ub_best_known`.
- `makespan_of_released_schedule`: makespan of the schedule in `census_best_schedules_10500.csv` (≤ `ub_best_known`).

**`data/census_best_schedules_10500.csv`**: `instance`, `makespan`, `source`, `inbound_start_times`, `outbound_start_times`. Start times are `;`-separated, in truck-index order. Every instance has a schedule, and every schedule was checked feasible with `cdc.check_schedule`. For 4 instances (m2_70_259, m4_70_225, u210_70_113, u24_70_18) the released schedule is better than the best-known UB reported in the paper (`ub_best_known`), because it comes from a rerun after the census; `makespan_of_released_schedule` gives its value.

**`data/census_upb_det_10500.csv`**: `upb_det` (deterministic UPB, forward), `upb_det_walk_exact` (1 if the capped walk was exact), `upb_det_cpu_s`, and the earlier bounds and HgR UB as above.

**`data/anytime_per_instance_70.csv`** (70 instances × k ∈ {1, 10, 100})
- Rival UBs and evaluation counts: `eacdc_restarts_*`, `eacdcml_code_*`, `hgr_*`.
- UBs of FB-ILS and iterated greedy read at each rival's evaluation count.
- 8 cells are empty: readings below a search's first schedule (filled by `fill_first_schedule.py` in the analysis).

**`data/instances_generated/<set>/*.dat`**: the benchmark text format, one file per instance. The generator seeds are in the folder names.

**`data/exact_fresh300_per_instance.csv`** (300 fresh instances, generator seed 20261193, 300 CPU s per method)
- `instance` (`<group>_<n1>_<index>`), `dock_group`, `n1`, `n2`, `m1`, `m2`; `lb_star` (LB*), `ub_start_fbils10s` (the warm-start schedule shared by all methods, FB-ILS 10 CPU s); `machine` (anonymised; all methods of an instance ran on the same machine).
- For each method `cp_uc`, `cp_lb`, `cp_plain`, `tim_mip` (CP-UC, CP-LB, CP-plain, time-indexed MIP): `_solved` (1 if optimality was proved), `_lb`, `_ub`, `_gap_pct` = 100 (ub − lb)/ub, `_cpu_s`. The time-indexed MIP ran on the 150 instances with n1 = 20 only; its cells are empty for n1 = 30. The scripts use short method identifiers; the docstring of `exact_fresh.py` maps them to these names.

**`data/tim_lp_strengthened_per_instance.csv`** (one row per instance and LP variant)
- `instance_set`: `small` (300 generated instances, n1 = 4–8), `real` (50 benchmark instances with n1 = 20), `large` (9 benchmark instances with n1 = 30–50).
- `horizon_T`; `upb` (our bound); `reference_ub` and `reference_kind` (the optimum, or the HgR UB on benchmark instances).
- `variant`: `base` (the original time-indexed LP) or the strengthened linking families `A`, `AC`, `ABC` (defined in `tim_lp_strong.py`); `lp_value` (empty if not solved within the limit), `status`, `cpu_s`, `cut_rounds`.

## Protocols
`PROTOCOL_HASHES.txt` lists one line per protocol: label, description, freeze date and SHA-256 digest. The protocols were frozen before the experiments; their SHA-256 digests are listed so the documents can be verified against this deposit; the documents are available from the author on request.

## Licence
- Code (`code/`, `env/`): MIT, see `LICENSE`.
- Data (`data/`): CC BY 4.0, see `LICENSE-DATA`. The column `ub_hgr_published` reproduces values published by Fonseca, Nogueira & Ravetti (2019).

## How to cite
```
Provezano Coutinho, F. (2026). Code and data for: Union-of-predecessors bounds and a forward–backward local search for multi-door cross-dock truck scheduling (Version 1.0.0) [Software and data]. Zenodo. https://doi.org/10.5281/zenodo.23933868
```
See also `CITATION.cff`. Please also cite Cota et al. (2016) for the benchmark. The HgR and EACDCml code here are our
re-implementations of the methods of Fonseca, Nogueira & Ravetti (2019) and Nogueira et al. (2024).
