# Prepare-only integration

Baseline source: `39e9c6fdb82611ecd7c05a0b0511e0ae4f41f269`.
Uploaded patch SHA256: `6a147d96e69bb42a77c8b16f5f30446d78c37f34a5e0ad59a690da8de7d7631c`.
Immutable model artifact: `7c198d6313364e96723f335c113e6275b1e1e070`,
manifest `b50a47304c2d647b34c6cb81a3df3addf19a6357011ea43ca7437792fd097578`.

Scope: register-local stable top8 selection, coalesced M1 ID publication and
removal of redundant all-SIMT metadata synchronization. Keep the existing
merged/no-bias/normalized-softmax admission: E256, top8, tokens1/2/4/8,
gate N1024 K512/2048, down N2048 K512. Preserve F16/BF16 arithmetic,
canonical weights, all GEMV choices, caller ABI and TC/mixed fallbacks.

Before packaging: host contracts and source-backed router tests; compile the
production execution image and the bounded router/prepare device gate. Reuse
unchanged TC, pack, prefill and paired GEMV images with original provenance.
Do not relabel an old source receipt as a fresh device result.

Box validation: IDs/weights against the unchanged shipping router, ties and
special values in the admitted domain, changed-input graph replay, M1 aliased
logits/weights, tokens1..8 and F16/BF16 controls. Timing follows correctness;
whole-model ABBA excludes each process's first complete pass, Asys is separate.

Returned unpatched model baseline: `kpack-q4-resume.7l1wtue2.results.tgz`,
SHA256 `528e996965ffc8215f4342ec0ff5ceb104b1499823b0c17bf9a061037156901b`.
TPOT6.609484ms vs native7.713883ms; prepare7.884423us in Asys.
The supplied 20% prepare gain remains externally reported until a matching
candidate device result returns. No fresh device claim from a host build.

Bounded handoff: stop at verified binaries and a reproducible box command;
no large sweep, model binary upload, or unrelated cleanup. Local build storage
is under /tmp because the data disk is nearly full; preserve existing files.
