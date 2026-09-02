# Q4 dense policy-v2 prebuilt

This execute-only PPU0010 payload was built from clean source
`425198f5d52377faf85eae4160cd44826e7f4388` with SDK
`2.1.1-a5c56e`.  Its manifest SHA-256 is
`3a009389b875adcb74b4d9271ffaa6bc65549dd6f728275cd88bc3928b3a5cc9`.

The payload contains one Q12 benchmark and its exact format-selected DSO:

```text
test_fq_kquant_layout_perf  b5e3797ead88026fbc3f16cd8848132211374f32002630911478136f662eb592
libquactlize_ppu.so         b4ba9abc245186c2879eeb820394f9b2390e46cf103318fa9a79e8345c6949c9
```

`evidence/` preserves the source/SDK authority, build log, CMake log, and
target build rule.  Their hashes are bound by `manifest.json` or reproduced
below:

```text
build-authority.json fb1a7f53d7ba62c7d94ad64468a9cd5b9645923315e9c9a1efb49f340d76f575
build.log           bd455c7683c1982844f7d4f75a75fbd4cedf572ea8f259c5b624e5c7d31209c7
cmake.log           69197b0ae348bac55f07f28ec8c23a2acec9c5b23f1b074d7d3f3ac3ffcc17c4
target-build.make   77774aa3de9fd060cdeaf2a3983f8513481ce53fdd1a7f20d77b7064ba7836a0
```

## Device execution

Use a clean source worktree at the exact build commit.  The artifact worktree
and result directory must be new strict children of `/workspace`.

```bash
SOURCE_COMMIT=425198f5d52377faf85eae4160cd44826e7f4388
ARTIFACT_BRANCH=artifacts/ppu0010/425198f-q12-policyv2-3a009389b875
REL=prebuilt/ppu0010/425198f/q12-policyv2-3a009389b875
ARTIFACT_WORKTREE=/workspace/quactlize-a04-artifact-425198f
SOURCE_WORKTREE=/workspace/quactlize-a04-source-425198f
OUT=/workspace/quactlize-a04-policy-v2-result-425198f

git fetch origin \
  "refs/heads/${ARTIFACT_BRANCH}:refs/remotes/origin/${ARTIFACT_BRANCH}"
git worktree add --detach "${ARTIFACT_WORKTREE}" "origin/${ARTIFACT_BRANCH}"
git -C "${ARTIFACT_WORKTREE}" lfs pull \
  --include="${REL}/bundle/test_fq_kquant_layout_perf,${REL}/bundle/libquactlize_ppu.so" \
  --exclude=""
git worktree add --detach "${SOURCE_WORKTREE}" "${SOURCE_COMMIT}"
git -C "${SOURCE_WORKTREE}" submodule update --init --recursive

BUNDLE="${ARTIFACT_WORKTREE}/${REL}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  3a009389b875adcb74b4d9271ffaa6bc65549dd6f728275cd88bc3928b3a5cc9
source "${PPU_SDK:?set PPU_SDK to the admitted SDK root}/envsetup.sh"

CUDA_VISIBLE_DEVICES=0 \
FQ_KQUANT_POLICY_V2_ROOT=/workspace \
FQ_KQUANT_POLICY_V2_BUNDLE="${BUNDLE}" \
PPU_SDK="${PPU_SDK}" \
PERF_ITERATIONS=11 \
PERF_WARMUPS=3 \
PERF_ROUNDS=3 \
OUT="${OUT}" \
  bash "${SOURCE_WORKTREE}/tools/run_fq_kquant_policy_v2_box.sh"
```

The runner must print both:

```text
FQ_KQUANT_POLICY_V2 verdict=PILOT_COMPLETE shapes=64 candidates=5
[fq-kquant-policy-v2-box] DIAGNOSTIC_COMPLETE ...
```

Those lines prove measurement completeness, not a shipping selector.  Run the
independent adjudicator from commit
`1417119ea02b9547d41ab8a9424047b8ea8418b5` (tool SHA-256
`add39cd8c559658523d8a44c27d30ac8983287bdcb18eb18f93e67ccb9b49059`)
against the result.  It emits only measured categorical leaves whose
conservative regret upper bound is at most 3%; gaps remain explicit.  This
pilot covers only N=1024, K=5120, M=1..64 and never authorizes a compiled
default outside that domain.
