"""Next bounded Q8 cohort: retain all three prior controls, no Cartesian sweep."""
from dev.gemv_simt.q8_dense_plan import inventory
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.q8_address_source import ARMS


def plan():
    points=[dict(point=i,**row) for i,row in enumerate(inventory())]
    return dict(schema='q8-address-window-v1',points=points,configs=configs(8),arms=list(ARMS),
        numeric_pairs=sum(len(p['numeric_m']) for p in points)*2*(len(ARMS)-1),
        timing_pairs=sum(len(p['timing_m']) for p in points)*2*(len(ARMS)-1),
        profiles=[dict(point=1,m=1,config=1),dict(point=4,m=8,config=0)],
        acu_reports=2*len(ARMS)*3,production_changed=False,score='MATCHED_CLOCK_CE_CYCLES',
        scope='Q8_SIGNED_ADDRESS_AND_LOAD_WINDOW_NOT_GLOBAL_SELECTION',
        frequency_policy='FRESH_1P7GHZ_CONTROLS_NO_1GHZ_RESCALING',
        omissions=['TWO_ROW_B_REUSE','SPLIT_K','MOE_CHANGES','MODEL_TPOT','PRODUCTION_ROUTING'])
