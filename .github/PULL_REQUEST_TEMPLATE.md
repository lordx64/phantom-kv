<!-- Thanks! Keep the invariants: zero base-weight modification, no refusal
     vector, deterministic scoreboard runs. Delete sections that don't apply. -->

## What

<!-- one paragraph -->

## Scoreboard evidence

<!-- REQUIRED for any change touching graft building, splicing, or eval:
     paste the aggregate numbers and attach artifacts/eval/run_<ts>.md -->

| arm | harmful refusals | harmless refusals | KL mean/max |
| --- | --- | --- | --- |
| before |  |  |  |
| after |  |  |  |

## Checks

- [ ] `phantom-eval --self-test` prints 8/8
- [ ] self-test still runs with no third-party dependencies (`pip install -e . --no-deps`)
- [ ] no weight modification, no refusal-direction computation anywhere
- [ ] README / docs updated if behavior or numbers changed
