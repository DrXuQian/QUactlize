from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dev.llama_ab.contract import SCOPE, coverage, family, inventory, judge, pilot, route_proof
from tools.run_kpack_llama_ab import point_request, record_load, save_record, runtime_payloads, identities, validate_loaded


def measured(ratio=.8):
    return dict(correctness='PASS',scope=SCOPE,rounds=[
        [dict(arm=a,event_us=[10*(ratio if a else 1)]*3) for a in (0,1,1,0)] for _ in range(6)])


class VerdictTest(unittest.TestCase):
    def test_no_five_percent_allowance(self):
        self.assertEqual(judge(measured(1.001))['status'],'REGRESSION')
        self.assertEqual(judge(measured(1))['status'],'PASS')
        self.assertEqual(judge(measured())['status'],'PASS')

    def test_overlap_is_not_pass(self):
        row=measured(.999)
        row['rounds'][0][1]['event_us']=[10.1]*3
        row['rounds'][0][2]['event_us']=[10.1]*3
        self.assertEqual(judge(row)['status'],'INCONCLUSIVE')

    def test_order_bias_is_inconclusive(self):
        row=measured()
        row['rounds'][0][3]['event_us']=[12]*3
        self.assertEqual(judge(row)['status'],'INCONCLUSIVE')

    def test_negative_controls(self):
        for change in ('correctness','scope','missing_round','order','sample_count','nan','zero'):
            with self.subTest(change=change):
                row=measured()
                if change in ('correctness','scope'):row[change]='WRONG'
                elif change=='missing_round':row['rounds'].pop()
                elif change=='order':row['rounds'][0][0]['arm']=1
                elif change=='sample_count':row['rounds'][0][0]['event_us'].pop()
                elif change=='nan':row['rounds'][0][0]['event_us'][0]=float('nan')
                elif change=='zero':row['rounds'][0][0]['event_us'][0]=0
                with self.assertRaises(ValueError):judge(row)

    def test_missing_does_not_pass(self):
        rows=[dict(id='a'),dict(id='b')]
        self.assertEqual(coverage(rows,{'a':dict(status='PASS')})['status'],'NOT_ADMITTED')
        self.assertEqual(coverage(rows,{k:dict(status='PASS') for k in ('a','b')})['status'],'PASS')
        for status in ('REGRESSION','INCONCLUSIVE','FAIL'):
            self.assertEqual(coverage(rows,{'a':dict(status='PASS'),'b':dict(status=status)})['status'],'NOT_ADMITTED')
        with self.assertRaises(ValueError):coverage(rows,{'c':dict(status='PASS')})

    def test_receipt_mutation_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'record.json'
            save_record(path,dict(authority={'device':'a'},status='PASS'))
            self.assertEqual(record_load(path,{'device':'a'})['status'],'PASS')
            with self.assertRaises(ValueError):record_load(path,{'device':'b'})
            row=json.loads(path.read_text());row['status']='REGRESSION';path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):record_load(path,{'device':'a'})


class InventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=inventory()

    def test_frozen_denominator(self):
        self.assertEqual(len(self.rows),1544)
        self.assertEqual(len({r['id'] for r in self.rows}),1544)
        self.assertEqual(Counter(r['workload']['operation'] for r in self.rows),dict(dense=810,indexed=734))
        self.assertEqual(len({family(r) for r in self.rows}),108)

    def test_profile_variants_survive_selector_promotions(self):
        rows=[r for r in self.rows if r['cohort']=='decode']
        self.assertEqual(len(rows),1114)
        self.assertEqual({r['workload']['router'] for r in rows},{'','real','spread','cluster','repeat2','repeat4'})

    def test_pilot_covers_all_interfaces(self):
        rows=pilot(self.rows)
        self.assertEqual(len(rows),32)
        self.assertEqual({(r['cohort'],r['workload']['operation'],r['workload']['qtype']) for r in rows},
                         {(r['cohort'],r['workload']['operation'],r['workload']['qtype']) for r in self.rows})
        from dev.llama_ab.contract import confirmed_prefill
        def routes(items):
            return {(r['workload']['operation'],r['workload']['qtype'],confirmed_prefill()[r['id']]['recipe']['route'])
                    for r in items if r['cohort']=='prefill'}
        self.assertEqual(routes(rows),routes(self.rows))

    def test_active_cold_read_set(self):
        row=next(r for r in self.rows if r['workload']['operation']=='indexed' and
                 r['workload']['m']==1)
        point=point_request(row,64*2**20)
        active=row['workload']['topk']
        self.assertEqual(point['ring']['active_experts'],active)
        self.assertGreaterEqual(point['ring']['active_read_bytes'],2.25*64*2**20)
        self.assertEqual(point['ring']['total_weight_bytes_per_arm'],
                         point['ring']['active_read_bytes']*row['workload']['experts']//active)
        self.assertEqual(len(point['owners']),active)

    def test_route_requires_compute_not_placement(self):
        row=next(r for r in self.rows if r['workload']['operation']=='indexed' and r['cohort']=='decode')
        w=row['workload'];name='ab_'+row['id'][:20]+'_0'
        text=f'[quactlize-plan] tensor={name} op=grouped route=gemv q={w["qtype"]} n={w["n"]} k={w["k"]} activation=BF16'
        self.assertTrue(route_proof(text,row))
        self.assertTrue(route_proof(text.replace('route=gemv','route=full-bf16').replace(' activation=BF16',''),row))
        for bad in (text.replace('BF16','FP16'),text.replace('op=grouped','op=dense'),
                    text.replace(f'n={w["n"]}','n=999'),text+'\n'+name+' native policy miss retain legacy',
                    f'[quactlize] {name} CUDA0_KPACK'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):route_proof(bad,row)

    def test_no_new_config_pool(self):
        for row in self.rows:
            self.assertNotIn('recipe',row)
            self.assertNotIn('candidates',row)

    def test_prefill_trace_cannot_use_old_route_or_helper(self):
        from dev.llama_ab.contract import confirmed_prefill
        for row in (r for r in self.rows if r['cohort']=='prefill'):
            w=row['workload'];r=confirmed_prefill()[row['id']]['recipe'];p=dict(r['parameters'])
            route='full-bf16' if r['kind']=='full' else r['route']
            text=(f'[quactlize-plan] tensor=ab_{row["id"][:20]}_0 op={"dense" if w["operation"]=="dense" else "grouped"} '
                  f'route={route} q={w["qtype"]} n={w["n"]} k={w["k"]} policy=19 channels=1 dequant={p["dequant_config"]} '
                  f'activation={"FP16" if w["compute"]=="f16" else "BF16"}')
            if r['kind']=='tc':text+=f' parent={p["symbol"]} split={p["split"]}'
            self.assertTrue(route_proof(text,row))
            for before,after in [('policy=19','policy=8'),('channels=1','channels=8'),
                                 (f'dequant={p["dequant_config"]}','dequant=99')]:
                with self.assertRaises(ValueError):route_proof(text.replace(before,after),row)


class SourceContractTest(unittest.TestCase):
    def test_merged_producer_private_dependencies_are_sealed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);build=root/'llama'
            libraries=[root/'public/libquactlize_ppu.so',
                root/'public/quactlize-runtime/libquactlize_kpack_dispatch.so',
                root/'bundle/libquactlize_kpack_dispatch.so',
                build/'bin/libggml-base.so',build/'bin/libggml-cuda.so']
            for p in libraries:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'HOST IDENTITY FIXTURE')
            manifest=root/'public/manifest.json';manifest.write_text('{}')
            args=SimpleNamespace(bundle=root/'bundle',legacy_bundle=None,llama_build=build,
                                 pack_library=libraries[0],jit_helper=None)
            # Actual public manifest/ELF verification has separate host tests.
            with patch('tools.build_ppu_library.verify') as check:
                paths=runtime_payloads(args);check.assert_called_once_with(root/'public')
            self.assertIn(libraries[1],paths)
            request=dict(candidate_runtime=identities(paths),carrier={'identity':{'libraries':identities(libraries[3:])}})
            data={'loaded_images':list(map(str,libraries))}
            self.assertTrue(validate_loaded(data,request))
            libraries[1].write_bytes(b'CHANGED DISPATCHER')
            with self.assertRaises(ValueError):validate_loaded(data,request)

    def test_native_dispatcher_and_eager_events(self):
        source=Path('dev/llama_ab/carrier.cpp').read_text()
        self.assertIn('ggml_backend_graph_compute_async',source)
        self.assertIn('ggml_mul_mat_id',source)
        self.assertIn('ggml_backend_cuda_buffer_type',source)
        self.assertIn('hggcEventCreateWithFlags',source)
        self.assertIn('rt.create(&start.context, 0)',source)
        self.assertNotIn('GraphLaunch',source)
        self.assertNotIn('__global__',source)
        self.assertIn('ggml_get_type_traits',source)
        self.assertIn('copies*traversals',source)


if __name__=='__main__':unittest.main()
