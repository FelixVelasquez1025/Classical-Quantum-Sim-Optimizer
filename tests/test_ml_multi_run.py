"""Cross-run validation grouping and deduplication without model fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ml.data import load_datasets


class MultiRunTests(unittest.TestCase):
    def dataset(self, root, name, family, digest, count=4):
        folder=root/name; folder.mkdir()
        entry=dict(path='c.qasm', family=family, normalized_sha256=digest, source_sha256=name, status='imported')
        (folder/'manifest.json').write_text(json.dumps({'circuits':[entry]}))
        config=dict(run_id=name, settings={'task':'auto','shots':1000}, normalization='v1', timeout_seconds=3600, feature_version=1)
        snapshot=dict(run_dir=str(folder),extension_sha256='native',provenance={'machine':'arm'},warnings=[])
        record=dict(path='c.qasm',family=family,normalized_sha256=digest,task='shots',features={'n':2},metadata=entry,rows={str(i):{} for i in range(count)})
        return dict(config=config,snapshot=snapshot,records=[record])

    def test_aliases_do_not_leak_across_sources_and_paths_are_namespaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.dataset(Path(tmp),'a','qft','hash1');b=self.dataset(Path(tmp),'b','qftentangled','hash2')
            with patch('ml.data.load_dataset',side_effect=[a,b]): d=load_datasets(['a','b'])
            self.assertEqual(len(d['records']),2)
            self.assertEqual(len({r['group'] for r in d['records']}),1)
            self.assertEqual({r['path'] for r in d['records']},{'a/c.qasm','b/c.qasm'})
            self.assertEqual(a['records'][0]['path'],'c.qasm')

    def test_duplicate_prefers_completeness_not_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.dataset(Path(tmp),'a','foo','same',2);b=self.dataset(Path(tmp),'b','bar','same',4)
            with patch('ml.data.load_dataset',side_effect=[a,b]): d=load_datasets(['a','b'])
            self.assertEqual(len(d['records']),1)
            self.assertEqual(d['records'][0]['source_run_id'],'b')
            self.assertEqual(d['snapshot']['removed_duplicate_observations'],1)

    def test_incompatible_settings_or_native_build_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.dataset(Path(tmp),'a','foo','one');b=self.dataset(Path(tmp),'b','bar','two')
            for change in ('settings','native','task'):
                bad=copy.deepcopy(b)
                if change=='settings':bad['config']['settings']['shots']=10
                if change=='native':bad['snapshot']['extension_sha256']='other'
                if change=='task':bad['records'][0]['task']='evolve'
                with patch('ml.data.load_dataset',side_effect=[a,bad]):
                    with self.assertRaises(ValueError):load_datasets(['a','b'])


if __name__=='__main__':unittest.main()
