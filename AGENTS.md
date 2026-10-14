# Repository work rules

- This is the Go1 deployment repository at /home/pi/yzWorkspace/sim2real_go1.
- Keep runtime entry points in scripts/, reusable control code in src/go1_sim2real/, and auxiliary tools in tools/ by purpose.
- Current robot configuration is config/robots/unitree5151.yaml. Retain calibration and safety settings unless a change is explicitly justified and verified.
- Model bundles in artifacts/ are immutable payloads: preserve checkpoints, manifests and training snapshots. Do not replace a bundle with a different policy.
- The runtime supports legacy 48/235 observations and the 19499 1072-dimensional elevation contract. 19499 is installed for offline validation; unknown-map hardware deployment remains unvalidated.
- For maintenance, run pytest and offline dry-run/preflight checks. Do not start motors or stop sport mode as part of code cleanup.
- After completing and validating an authorized change, commit it and push to GitHub promptly. If push fails, report the pending commit and reason. Never force-push or overwrite unrelated work.
- Never store SSH passwords, tokens or private keys in the repository.
