# codex status

    updated-at:     2026-08-14 05:17:01 UTC
    inbox-consumed: 169
    working-on:     fixed the ACU artifact contract: site ACU appends .acurep to -o, so the runner now resolves base vs base.acurep and requires exactly one rather than declaring a produced report missing
    blocked-on:     clean BPC1/2/3 subject-only ACU reports; the existing BPC1 .acurep is valid but the aborted bundle did not reach BPC2/3
    local-gates:    runner bash syntax PASS; standalone target contract PASS with 19/19 structural plants including ambiguous report resolution; composed Marlin contract PASS
    last-commit:    Accept the site ACU report suffix explicitly
    last-heartbeat: 306
