import inspect
import json
import sys
import types


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Mask:
    class _Device:
        type = "cpu"

    device = _Device()

    def __init__(self, tokens):
        self.tokens = tokens

    def sum(self):
        return _Scalar(self.tokens)


def _fake_transformers():
    package = types.ModuleType("transformers")
    trainer_module = types.ModuleType("transformers.trainer")

    class TrainerCallback:
        pass

    class Optimizer:
        def __init__(self):
            self.steps = 0
            self.zeros = 0

        def step(self, closure=None):
            self.steps += 1
            return closure() if closure is not None else None

        def zero_grad(self, set_to_none=False):
            self.zeros += 1
            return set_to_none

    class Scheduler:
        def __init__(self):
            self.steps = 0

        def step(self, epoch=None):
            self.steps += 1
            return epoch

    class GradientState:
        sync_gradients = False

    class Accelerator:
        def __init__(self):
            self.gradient_state = GradientState()
            self.backward_calls = 0

        def backward(self, loss, **kwargs):
            self.backward_calls += 1
            return loss, kwargs

    class State:
        global_step = 0
        num_input_tokens_seen = 0

    class Trainer:
        def __init__(self):
            self.state = State()
            self.accelerator = Accelerator()
            self.optimizer = Optimizer()
            self.lr_scheduler = Scheduler()
            self.model = types.SimpleNamespace()
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def get_batch_samples(self, iterator, count, device):
            del device
            return [next(iterator) for _ in range(count)], None

        def _prepare_inputs(self, inputs):
            return inputs

        def compute_loss(self, model, inputs, **kwargs):
            del model, inputs, kwargs
            return 1.0

        def training_step(self, model, inputs, num_items_in_batch=None):
            prepared = self._prepare_inputs(inputs)
            loss = self.compute_loss(
                model, prepared, num_items_in_batch=num_items_in_batch
            )
            self.accelerator.backward(loss)
            return loss

        def _clip_grad_norm(self, model):
            del model
            return 2.0

        def _get_grad_norm(self, model, grad_norm=None):
            del model
            return 2.0 if grad_norm is None else grad_norm

        def _maybe_log_save_evaluate(self, *args, **kwargs):
            del args, kwargs
            return None

        def create_optimizer(self):
            return self.optimizer

        def create_scheduler(self, num_training_steps=None, optimizer=None):
            del num_training_steps, optimizer
            return self.lr_scheduler

    def update_kt_lora_pointers(model):
        model.updated = getattr(model, "updated", 0) + 1

    package.Trainer = Trainer
    package.TrainerCallback = TrainerCallback
    package.trainer = trainer_module
    trainer_module.Trainer = Trainer
    trainer_module.update_kt_lora_pointers = update_kt_lora_pointers
    return package, trainer_module, Trainer, Optimizer, Scheduler


def test_gas2_timeline_is_reversible_and_uses_trainer_tokens(tmp_path, monkeypatch):
    package, trainer_module, trainer_class, _optimizer_class, scheduler_class = (
        _fake_transformers()
    )
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.trainer", trainer_module)
    monkeypatch.setenv("KT_STEP_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("KT_STEP_PROFILE_RUN_ID", "integration")
    monkeypatch.setenv("KT_STEP_PROFILE_CUDA_EVENTS", "0")
    monkeypatch.setenv("KT_STEP_PROFILE_BOUNDARY_MEMORY", "0")
    monkeypatch.setenv("KT_STEP_PROFILE_HOST_INTERVAL_MS", "1000")
    monkeypatch.setenv("KT_STEP_PROFILE_NVML_INTERVAL_MS", "1000")

    from scripts.sft_step_profile import trainer_integration

    original_training_step = trainer_class.training_step
    original_scheduler_step = scheduler_class.step
    manager = trainer_integration.install("phase")
    trainer = trainer_class()

    class PreparedOptimizer:
        def __init__(self, inner):
            self.inner = inner

        def step(self, closure=None):
            return self.inner.step(closure=closure)

        def zero_grad(self, set_to_none=False):
            return self.inner.zero_grad(set_to_none=set_to_none)

    raw_optimizer = trainer.optimizer
    trainer.optimizer = PreparedOptimizer(raw_optimizer)
    original_optimizer_signature = inspect.signature(trainer.optimizer.zero_grad)
    callback = trainer.callbacks[0]
    callback.on_train_begin(None, trainer.state, None, model=trainer.model)

    batches, _ = trainer.get_batch_samples(
        iter([{"attention_mask": _Mask(3)}, {"attention_mask": _Mask(4)}]), 2, None
    )
    trainer.accelerator.gradient_state.sync_gradients = False
    trainer.training_step(trainer.model, batches[0])
    trainer.accelerator.gradient_state.sync_gradients = True
    trainer.training_step(trainer.model, batches[1])
    grad_norm = trainer._clip_grad_norm(trainer.model)
    trainer._get_grad_norm(trainer.model, grad_norm)
    trainer.optimizer.step()
    trainer_module.update_kt_lora_pointers(trainer.model)
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=False)
    trainer.state.global_step = 1
    trainer.state.num_input_tokens_seen = 7
    trainer._maybe_log_save_evaluate()
    callback.on_train_end(None, trainer.state, None)
    manager.close()

    summary = json.loads((tmp_path / "rank_0" / "step_summary.json").read_text())
    step = summary["steps"][0]
    assert step["status"] == "ok"
    assert step["microbatch_count"] == 2
    assert step["observed_tokens"] == 7
    assert step["tokens"] == 7
    assert step["phases"]["forward"]["calls"] == 2
    assert step["phases"]["backward"]["calls"] == 2
    assert step["phases"]["grad_clip"]["calls"] == 1
    assert step["phases"]["optimizer"]["calls"] == 1
    assert step["phases"]["kt_post_update"]["calls"] == 1
    assert step["phases"]["zero_grad"]["calls"] == 1
    assert step["accounting_error_ns"] == 0
    assert trainer.model.updated == 1
    assert raw_optimizer.steps == 1
    assert raw_optimizer.zeros == 1

    assert trainer_class.training_step is original_training_step
    assert scheduler_class.step is original_scheduler_step
    assert (
        inspect.signature(trainer.optimizer.zero_grad) == original_optimizer_signature
    )
    assert "step" not in trainer.optimizer.__dict__
    assert "zero_grad" not in trainer.optimizer.__dict__
    assert "_prepare_inputs" not in trainer.__dict__
    assert "compute_loss" not in trainer.__dict__
    assert "backward" not in trainer.accelerator.__dict__
    assert trainer.callbacks == []
    source = (tmp_path / "rank_0" / "phase_events.jsonl").read_text()
    assert "synchronize" not in source
    markers = [
        json.loads(line)
        for line in source.splitlines()
        if json.loads(line).get("record_type") == "microbatch_marker"
    ]
    assert [row["microbatch"] for row in markers] == [0, 1]


def test_failed_forward_is_error_scope_and_partial_step(tmp_path, monkeypatch):
    package, trainer_module, trainer_class, _optimizer_class, _scheduler_class = (
        _fake_transformers()
    )
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.trainer", trainer_module)
    monkeypatch.setenv("KT_STEP_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("KT_STEP_PROFILE_RUN_ID", "failure")
    monkeypatch.setenv("KT_STEP_PROFILE_CUDA_EVENTS", "0")
    monkeypatch.setenv("KT_STEP_PROFILE_BOUNDARY_MEMORY", "0")
    monkeypatch.setenv("KT_STEP_PROFILE_HOST_INTERVAL_MS", "1000")
    monkeypatch.setenv("KT_STEP_PROFILE_NVML_INTERVAL_MS", "1000")

    from scripts.sft_step_profile import trainer_integration

    manager = trainer_integration.install("phase")
    trainer = trainer_class()

    def fail_loss(*args, **kwargs):
        del args, kwargs
        raise LookupError("intentional forward failure")

    trainer.compute_loss = fail_loss
    callback = trainer.callbacks[0]
    callback.on_train_begin(None, trainer.state, None, model=trainer.model)
    batches, _ = trainer.get_batch_samples(
        iter([{"attention_mask": _Mask(2)}]), 1, None
    )
    try:
        trainer.training_step(trainer.model, batches[0])
    except LookupError as exc:
        assert str(exc) == "intentional forward failure"
    else:
        raise AssertionError("training_step swallowed the original failure")
    manager.close(partial=True)

    rows = [
        json.loads(line)
        for line in (tmp_path / "rank_0" / "phase_events.jsonl")
        .read_text()
        .splitlines()
    ]
    forward = next(row for row in rows if row.get("name") == "forward")
    assert forward["status"] == "error"
    assert forward["error_type"] == "LookupError"
    summary = json.loads((tmp_path / "rank_0" / "step_summary.json").read_text())[
        "steps"
    ][0]
    assert summary["status"] == "partial"
    assert trainer.compute_loss is fail_loss
    assert trainer.callbacks == []
