"""Native trainer with isolated storage-only save override."""
import os
import hydra
import ray
from verl.trainer import main_ppo
from verl.utils.device import auto_set_device
from verl.experimental.reward_loop import migrate_legacy_reward_impl


class StagedTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        from native_trainer import install_native_trainer
        from staged_checkpoint import install
        install_native_trainer()
        install()
        return super().run(config)


@hydra.main(config_path=os.path.join(os.environ.get('VERL_ROOT', os.path.expanduser('~/verl')), 'verl/trainer/config'), config_name='ppo_trainer', version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(StagedTaskRunner))


if __name__ == '__main__':
    main()
