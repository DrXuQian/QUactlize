"""Address, ownership, pruning and fail-closed evidence checks without a GPU."""
import csv
import io
import unittest

from dev.gemv_simt.q8_topology import POINTS,CONFIGS,GEOMETRIES,inventory,config,access,source
from dev.gemv_simt.model_followup import kernel_body
from dev.gemv_simt.run_q8_topology import choose_finalists,validate_profile,summarize


class Topology(unittest.TestCase):
    def test_bounded_inventory_and_partition_coverage(self):
        self.assertEqual(len(GEOMETRIES),13)
        self.assertEqual(len(CONFIGS),26)
        for (n,k,_),expected in zip(POINTS,(110,30,150)):
            plan=inventory(n,k);self.assertEqual(len(plan['cells']),expected)
            self.assertEqual(len({c['key'] for c in plan['cells']}),expected)
            covered={(c['recipe'],c['split']) for c in plan['cells']}
            pruned={(c['recipe'],c['split']) for c in plan['pruned']}
            self.assertFalse(covered&pruned)
            self.assertEqual(len(covered|pruned),26*4)
            for cell in plan['cells']:
                c=config(cell);workers=c.warps*32//c.columns
                groups=[g for partition in range(c.split) for worker in range(workers)
                        for g in range(partition*workers+worker,k//32,c.split*workers)]
                self.assertEqual(sorted(groups),list(range(k//32)))
                self.assertLessEqual(workers*c.split,k//32)
                self.assertEqual(n%c.tile_n,0)
                self.assertIn(c.tile_n,(8,16,32))
            for row in plan['pruned']:
                c=CONFIGS[row['recipe']]
                self.assertGreater(c.warps*32//c.columns*row['split'],k//32)

    def test_cooperative_activation_exact_owners(self):
        for c in CONFIGS:
            count=32//c.columns
            self.assertIn(count,(2,4,8)) # exact load2/load4/load8 dispatch
            for lane in range(32):
                for offset in range(0,32,2):
                    owner=(lane&~(c.columns-1))+offset//count
                    index=offset%count
                    self.assertEqual(owner//c.columns,lane//c.columns)
                    self.assertEqual((owner%c.columns)*count+index,offset)
                    self.assertLess(index+1,count)

    def test_warp_reduce_scatter_and_cta_fold(self):
        # Symbolic sums model the exact XOR exchange and first-warp fold.
        # Every output must contain all K workers once, including C16/P2.
        for c in CONFIGS:
            partial=[]
            for warp in range(c.warps):
                values=[[{((lane%c.columns)*c.values+p,(warp*32+lane)//c.columns)}
                         for p in range(c.values)] for lane in range(32)]
                count,stride=c.values,c.columns
                while count>1:
                    before=values
                    values=[[before[lane][2*i+(1 if lane&stride else 0)] |
                             before[lane^stride][2*i+(0 if (lane^stride)&stride else 1)]
                             for i in range(count//2)] for lane in range(32)]
                    count//=2;stride*=2
                while stride<32:
                    before=values;values=[[before[l][0]|before[l^stride][0]] for l in range(32)];stride*=2
                slots={}
                for lane in range(c.tile_n):
                    slot=(lane%c.columns)*c.values+lane//c.columns
                    self.assertNotIn(slot,slots);slots[slot]=values[lane][0]
                partial.append(slots)
            fold=[]
            stripes=32//c.tile_n
            for lane in range(32):
                folded=set()
                for warp in range(lane//c.tile_n,c.warps,stripes):folded|=partial[warp][lane%c.tile_n]
                fold.append(folded)
            stride=c.tile_n
            while stride<32:
                before=fold;fold=[before[l]|before[l^stride] for l in range(32)];stride*=2
            for lane in range(c.tile_n):
                self.assertEqual(fold[lane],{(lane,w) for w in range(c.warps*32//c.columns)})

    def test_request_footprints_all_planes(self):
        for columns,values,util64 in ((4,2,.25),(4,4,.5),(8,4,1.),(16,2,1.)):
            cell=next(x for x in inventory(512,2048)['cells'] if
                      config(x).columns==columns and config(x).values==values and config(x).variant==5)
            p=access(cell,512,2048)
            self.assertEqual(p['streams'][0]['footprint']['64']['unique_sector_utilization'],util64)
            self.assertTrue(any(x['name'].startswith('A-') for x in p['streams']))
            self.assertTrue(any(x['name'].startswith('metadata-') for x in p['streams']))
            for stream in p['streams']:
                self.assertEqual(set(stream['footprint']),{'32','64','128'})
            weak=access(cell,512,2048,dict(A=0,low=0,high=0,units=2))
            self.assertFalse(weak['metadata_vectorized'])

    def test_unchanged_decoder_and_full_scope(self):
        text=source()
        self.assertEqual(text.count(kernel_body(True)),1)
        self.assertEqual(text.count('q8_topology::kernel<'),26)
        self.assertIn('uintptr_t(c->output)|uintptr_t(c->workspace)',text)
        for split in (2,4,8):self.assertIn(f'reduce_decode<{split}>',text)
        self.assertIn('simt::register_reuse_reduce<8>',text)
        self.assertIn('c->rows!=1',text)
        self.assertNotIn('atomic',text)

    def test_finalists_keep_shipping_s1_split_and_c16(self):
        cells=inventory(512,2048)['cells']
        rows=[dict(cell=c,status='PASS',samples_us=[10+i]*5) for i,c in enumerate(cells)]
        # A bogus faster numeric failure can never win.
        rows[0].update(status='FAIL',samples_us=[.001]*5)
        keys=choose_finalists(rows)
        self.assertEqual(keys[:2],['shipping','clone'])
        self.assertNotIn(cells[0]['key'],keys)
        selected=[config(c) for c in cells if c['key'] in keys]
        self.assertTrue(any(c.split==1 for c in selected))
        self.assertTrue(any(c.split>1 for c in selected))
        self.assertTrue(any(c.columns==16 for c in selected))
        with self.assertRaises(ValueError):choose_finalists([dict(status='FAIL')])

    def test_acu_exact_kernel_and_full_reducer(self):
        cells=inventory(512,2048)['cells'];cell=next(c for c in cells if c['split']==4 and c['reducer']==1)
        c=config(cell)
        producer=f'void quactlize::execution::q8_topology::kernel<1, 0, {c.variant-4}, {c.columns}, {c.warps}, {c.values}>(qkg_call_v1, int)'
        rows=[dict(ID=0,**{'Kernel Name':producer,'Kernel Mangled Name':'producer',
                           'Grid Size':f'({4*512//c.tile_n}, 1, 1)','Block Size':f'({c.warps*32}, 1, 1)'}),
              dict(ID=1,**{'Kernel Name':'void quactlize::decode::reduce_decode<4, float>(float const*, float*, int)',
                           'Kernel Mangled Name':'reducer','Grid Size':'(8, 1, 1)','Block Size':'(32, 1, 1)'})]
        def raw(records):
            out=io.StringIO();writer=csv.DictWriter(out,fieldnames=rows[0],quoting=csv.QUOTE_ALL)
            writer.writeheader();writer.writerows(records);return out.getvalue()
        self.assertEqual(len(validate_profile(raw(rows),512,2048,cell['key'],cells)),2)
        with self.assertRaises(ValueError):validate_profile(raw(rows[:1]),512,2048,cell['key'],cells)
        with self.assertRaises(ValueError):validate_profile(raw(rows*2),512,2048,cell['key'],cells)
        rows[0]['Block Size']='(32,1,1)'
        with self.assertRaises(ValueError):validate_profile(raw(rows),512,2048,cell['key'],cells)

    def test_confirmation_uses_full_six_round_distributions(self):
        samples={'shipping':[[10.]*15 for _ in range(6)],'candidate':[[9.]*15 for _ in range(6)]}
        result=summarize(samples,27000000,40)
        self.assertAlmostEqual(result['candidate']['delta_pct'],-10.)
        self.assertAlmostEqual(result['shipping']['effective_weight_MBU_pct'],100.)
        samples['candidate'][0][0]=float('nan')
        with self.assertRaises(ValueError):summarize(samples,1,40)
        samples['candidate']=[[1.]*15 for _ in range(5)]
        with self.assertRaises(ValueError):summarize(samples,1,40)
        with self.assertRaises(ValueError):summarize({'candidate':[[1.]*15]*6},1,40)


if __name__=='__main__':unittest.main()
