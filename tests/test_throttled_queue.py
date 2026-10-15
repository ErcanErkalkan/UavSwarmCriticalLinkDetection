import json, os, sys, tempfile, subprocess, unittest, textwrap
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
Q=BASE/'run_acld_p0_6_throttled_queue.py'

class ThrottledIntegrationTests(unittest.TestCase):
    def test_a2_c1_two_and_three_active(self):
        with tempfile.TemporaryDirectory(prefix='acld_throttle_test_') as td:
         p=Path(td)
         for name in ['acld_p0_6_common.py','acld_full_experimental_campaign_P0_4_corrected.ipynb']:
          (p/name).write_text('stub')
         (p/'run_acld_p0_6_unit_worker.py').write_text(textwrap.dedent('''
        import argparse,json,time
        from pathlib import Path
        p=argparse.ArgumentParser();p.add_argument('--project-root');p.add_argument('--campaign');p.add_argument('--worker-index',type=int);p.add_argument('--workers',type=int);a=p.parse_args();root=Path(a.project_root)
        if a.campaign == 'C2':
         d=root/'acld_full_experiment_outputs_P0_6'/'C2_runtime_scaling';d.mkdir(parents=True,exist_ok=True);(d/'runtime_scaling_full.csv').write_text('n,ms\\n50,1\\n');raise SystemExit(0)
        wd=root/'acld_full_experiment_outputs_P0_6_workers'/('A2_detection_rician' if a.campaign=='A2' else 'C1_accuracy_scaling')/f'worker_{a.worker_index:02d}_of_06'
        time.sleep(.1)
        (wd/'_worker_status.json').write_text(json.dumps(dict(campaign=a.campaign,worker_index=a.worker_index,workers=6,assigned_units=1,completed_or_reused_units=1)))
        '''))
         (p/'merge_acld_p0_6_campaign.py').write_text(textwrap.dedent('''
        import argparse,json
        from pathlib import Path
        p=argparse.ArgumentParser();p.add_argument('--project-root');p.add_argument('--campaign');p.add_argument('--workers');a=p.parse_args()
        d=Path(a.project_root)/'acld_full_experiment_outputs_P0_6'/('A2_detection_rician' if a.campaign=='A2' else 'C1_accuracy_scaling');d.mkdir(parents=True,exist_ok=True);(d/'_merge_report.json').write_text(json.dumps(dict(gate='PASS')))
        '''))
         for name in ['B1_closed_loop_nominal','B2_closed_loop_stress','C3_gcs_sensitivity']:
          d=p/'acld_full_experiment_outputs_P0_6'/name;d.mkdir(parents=True)
          (d/'_merge_report.json').write_text(json.dumps(dict(gate='PASS')))
         for name in ['A2_detection_rician','C1_accuracy_scaling']:
          for idx in range(1,7):
           d=p/'acld_full_experiment_outputs_P0_6_workers'/name/f'worker_{idx:02d}_of_06';d.mkdir(parents=True)
           (d/'worker_manifest.json').write_text(json.dumps(dict(workers=6,worker_index=idx)))
           (d/'detection_episode_metrics_long.csv').write_text('seed,scenario,method\n')
         env=os.environ.copy();env['PYTHONPATH']=str(BASE)
         for cap in [2,3]:
          (p/'acld_full_experiment_outputs_P0_6'/'C2_runtime_scaling'/'runtime_scaling_full.csv').unlink(missing_ok=True)
          result=subprocess.run([sys.executable,str(Q),'--project-root',str(p),'--active-per-campaign',str(cap),'--poll-seconds','.1'],env=env,capture_output=True,text=True,timeout=30)
          assert result.returncode==0, result.stderr
          status=json.loads((p/'acld_p0_6_logs_PARALLEL'/'QUEUE_STATUS.json').read_text())
          assert status['state']=='WAITING_FOR_A1_FINALIZATION' and status['c2_pass']
          assert status['active_workers_per_campaign']==cap
          assert len(status['completed_shards']['A2'])==6
          assert len(status['completed_shards']['C1'])==6
          assert not (p/'acld_p0_6_logs_PARALLEL'/'THROTTLED_QUEUE.lock').exists()
          print('integration PASS for',cap,'active workers per campaign')

if __name__ == "__main__":
    unittest.main()
