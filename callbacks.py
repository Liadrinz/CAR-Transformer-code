from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from modeling import CrossAttentionReinforcing


class MixerCallback(TrainerCallback):
    
    def __init__(self, init_pg_weight=1.0) -> None:
        super().__init__()
        self.init_pg_weight = init_pg_weight
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: CrossAttentionReinforcing, **kwargs):
        step = self.init_pg_weight / args.num_train_epochs
        model.pg_weight = max(0, model.pg_weight - step)
