"""Resume must preserve the original budget and reject silent rollback."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_physx_brax as trainer


def setup_run(tmp_path):
    spec = {'ppo': {'num_parallel_envs': 384, 'unroll_length': 64},
            'requested_training_transitions': 9977856, 'initialization': 'fresh'}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(tmp_path/'declaration.json', spec)
    for step in (24576, 49152, 73728):
        folder=tmp_path/'checkpoints'/f'transition_{step:08d}';folder.mkdir(parents=True)
        (folder/'learner_state.pkl').write_bytes(b'opaque learner bytes')
        (folder/'actor.npz').write_bytes(b'opaque actor bytes')
        write(folder/'identity.json', {'training_transitions': step,
              'learner_state_sha256': hashlib.sha256(b'opaque learner bytes').hexdigest(),
              'actor_sha256': hashlib.sha256(b'opaque actor bytes').hexdigest()})
        write(folder/'export_verification.json', {'passed': True})
        write(tmp_path/'development'/folder.name/'result.json', {'complete': True, 'physics_transitions': 512})
    write(tmp_path/'best_model.json', {'checkpoint': str(tmp_path/'checkpoints/transition_00024576'), 'training_transitions': 24576, 'development_mean_return': 120})
    write(tmp_path/'status.json', {'stage': 'sampling', 'training_transitions': 74112, 'update': 3})
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps({'training_transitions': n, 'update': n//24576})+'\n' for n in (24576,49152,73728)))
    return spec, tmp_path/'checkpoints/transition_00073728'


def test_resume_keeps_budget_latest_weights_and_earlier_best(tmp_path):
    spec, checkpoint=setup_run(tmp_path)
    context=trainer.resume_context(tmp_path, checkpoint, spec)
    assert context['start_total']==73728
    assert context['start_update']==3
    assert context['remaining_transitions']==9904128
    assert context['best']['training_transitions']==24576
    assert context['evaluation_physics_transitions']==1536
    assert context['discarded_recorded_transitions']==384
    assert context['exact_environment_resume'] is False


@pytest.mark.parametrize('bad', ['hash','config','rollback'])
def test_resume_rejects_corruption_config_change_and_older_checkpoint(tmp_path,bad):
    spec,checkpoint=setup_run(tmp_path)
    if bad=='hash':(checkpoint/'learner_state.pkl').write_bytes(b'broken')
    if bad=='config':spec['requested_training_transitions']+=24576
    if bad=='rollback':checkpoint=tmp_path/'checkpoints/transition_00049152'
    with pytest.raises(ValueError):trainer.resume_context(tmp_path,checkpoint,spec)
