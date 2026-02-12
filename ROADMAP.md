# Roadmap

## Phase 1: Foundation (now)

- Keep all experiments runnable on low-resource machines.
- Define core metrics:
  - quality (task accuracy)
  - accessibility (cost per run, memory footprint, latency)
  - decentralization (number/diversity of contributors and nodes)
- Publish reproducible baselines in this repo.
- Enforce baseline regression checks in CI.

## Phase 2: Distributed learning on commodity hardware

- Extend the current FedAvg baseline with:
  - non-IID robustness tests
  - client drop-out simulation
  - secure-aggregation mock flow
- Track communication compression effects (int8 and sparse updates).

## Phase 3: Small-model utility layer

- Add CPU-optimized small-model tasks:
  - local classification
  - lightweight retrieval-augmented generation
- Optimize for low-RAM environments.
- Record energy/time/cost benchmarks.

## Phase 4: Governance and openness

- Document model/data provenance.
- Define transparent contribution and release policy.
- Adopt a clear open license and public issue/decision process.

## Phase 5: Public network pilot

- Deploy a small volunteer node network.
- Measure:
  - uptime
  - fairness across node capabilities
  - quality vs centralized baseline
- Publish periodic open reports.
