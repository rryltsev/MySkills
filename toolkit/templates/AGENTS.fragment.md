# MLIP cluster workflows

User-specific settings are in `.mlip/config.json`; do not assume old account paths or budgets.
Matching skills are in `.agents/skills/`. Start migration with `umt-hpc` and the offline doctor.
Read `.mlip/docs/00-start.md` and the relevant workflow reference before cluster work.
Run compute/install/build work under Slurm on storage verified writable from compute nodes.
Persist unique run IDs, immutable inputs/hashes, submission intent, scheduler IDs and actual outputs.
Reconcile uncertain submissions before retrying. Validate scientific outputs separately from scheduler status.
Keep DFT and teacher labels separate; prevent grouped split leakage; preserve stress/units and model hashes.
Historical `.source` recipes are reference material and require new-user adaptation.

Before a new UMT run or batch, ask the user to select the login unless they already selected it for that batch. Check the matching Slurm association, resources and authorized budget.
For repeated DPA4 evaluation, phonons and MD, freeze with `DP_TRITON_INFER=2` at export time. Verify energy, forces and stress against the source checkpoint, then load the accepted frozen `.pt2` in the actual calculation. Keep `.pt` for training/restart and explicit reference diagnostics. Investigate an export or validation failure before proceeding.
