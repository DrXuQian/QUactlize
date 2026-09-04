# Exhaustive K-pack discovery artifact

This artifact contains the complete prebuilt-only PPU campaign generated from
source `ec67811bd709eace941daf3c650d45df574b1a87`: 64 independently verified
partition artifacts, 2,216 binary shards, 70,618 compiled parents, the
1,381-cell workload authority, and a frozen eight-worker assignment covering
340,282 binary-shard/workload atoms.

The three `campaign.tar.zst.part-*` files are Git LFS objects. The archive
expands to `campaign-ec67811/` and includes the distributed catalog and all
payloads. It does not contain a compiler and the launcher never invokes one.

Use runner commit `5919afa07d57ecb21bc2a5c73ce5b78f5c929648` or reject the
artifact. That runner is required for the proved structural-only FQ shard and
binds each payload's exact runtime-library subset to the full probe closure.

After hydrating this directory with `git lfs pull`, verify and extract:

```bash
sha256sum -c campaign.parts.sha256
cat campaign.tar.zst.part-* > /workspace/campaign-ec67811-box8.tar.zst
echo 'a1b61b7938542f2eaeec8394b23d3ced89727119701c261280475fe60478261f  /workspace/campaign-ec67811-box8.tar.zst' |
  sha256sum -c -
zstd -dc /workspace/campaign-ec67811-box8.tar.zst | tar -xf - -C /workspace
```

Then set `PPU_SDK` to the compatible SDK root and run:

```bash
QUACTLIZE_ROOT=/workspace/quactlize-runner-5919afa \
CAMPAIGN=/workspace/campaign-ec67811 \
RUN=/workspace/kpack-discovery-ec67811 \
bash run_box8.sh
```

For the exhaustive timing census, set `CORRECTNESS_REPEATS=1`; run the
256-repeat finalist and 8,192-repeat shipping stability gates only after the
timing census has selected their much smaller denominators. The default stays
at 256 so an omitted variable cannot silently weaken an established run.

The launcher validates the catalog and assignment, hashes every assigned
payload, probes and proves eight homogeneous devices, and starts one resumable
prebuilt worker per device with `nohup`.
