"""Keep implementation-family claims separate from mixed-provider winners."""

FAMILIES = ('all', 'machete-streamk')


def matches_family(arm, config_detail, family):
    if family not in FAMILIES:
        raise ValueError('unknown comparison family')
    if family == 'all':
        return True
    # The descriptor comes from the compiled module, not the selected row ID.
    # A Machete reader with a Marlin scheduler is NOT Machete/Stream-K parity.
    return (arm in ('machete', 'quactlize')
            and '_streamk' in config_detail
            and 'marlin' not in config_detail
            and (arm == 'machete' or '_prepacked_' in config_detail))


def filter_rows(rows, family):
    selected = [row for row in rows
                if matches_family(row['arm'], row.get('config_detail', ''), family)]
    if family == 'machete-streamk' and {row['arm'] for row in selected} != {'machete', 'quactlize'}:
        raise ValueError('Machete/Stream-K comparison requires both pure families')
    return selected
