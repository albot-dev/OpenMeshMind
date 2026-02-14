# Volunteer Data Submission

Volunteer data contributions are optional right now, but always welcome.
If you run the onboarding script, you can open a pull request with your generated bundle whenever you want.

## 1. Generate onboarding data bundle

From your clone:

```bash
bash scripts/volunteer_node_setup.sh --node-id volunteer-node-001
```

Expected output:

- `pilot/submissions/<node_id>_onboarding_<timestamp>.tgz`

## 2. Prepare a PR branch

```bash
git checkout -b volunteer-data-volunteer-node-001
```

## 3. Add the bundle to git

`pilot/submissions/` is gitignored, so add with force:

```bash
git add -f pilot/submissions/<node_id>_onboarding_<timestamp>.tgz
git commit -m "pilot: add onboarding bundle for <node_id>"
git push origin volunteer-data-volunteer-node-001
```

## 4. Open pull request

Open a PR to `main` and include:

- node id
- run date (UTC)
- hardware tier / region / network tier used
- any notable errors or retries during onboarding

Maintainers can import submitted bundles with:

```bash
python3 scripts/solo_multi_machine_mode.py \
  --bundles-glob 'pilot/submissions/*_onboarding_*.tgz' \
  --min-nodes 1 \
  --min-passed 0 \
  --require-metrics-files
```

## Notes

- Do not commit your `.env`.
- The generated node config stores `token_env_var` key name, not the token value.
- If you want to contribute code improvements instead of data, open a regular PR.
