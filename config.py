import json
import os

class Config:
    def __init__(self, config_path=None):
        # Default Configuration
        self.defaults = {
            "model": {
                "embed_dim": 32,
                "dropout": 0.1,
                "num_layers": 2,
                "num_heads": 5,
                "attn_bias_mode": "sum",
                "prenorm": True,
                "model_type": "hdg_moe" # Options: "stan", "hdg_moe"
            },
            "moe": {
                "mechanism_groups": 4,
                "experts_per_group": [2, 3, 4, 3], # Total 12 experts
                "gate1_topk": 2,
                "gate2_topk": 1,
                "expert_hidden_mult": 4.0,
                "gate_hidden_dim": 128,
                "gate_dropout": 0.0,
                "gate_temperature": 1.0,
                "router_eps": 1e-9
            },
            "loss": {
                "lambda_lb1": 0.01,
                "lambda_lb2": 0.01,
                "lb_type": "l2_uniform",
                "ignore_padding_in_lb": True
            },
            "data": {
                "use_category": False,
                "use_admin": False,
                "category_vocab_size": None, # Infer or set
                "admin_vocab_size": None,
                "region_inject_gamma_init": 0.1
            },
            "train": {
                "lr": 3e-3,
                "num_epoch": 100,
                "batch_size": 6,
                "num_neg": 10,

                "patience": 10,
                "device": "cuda",
                "log_dir": "log"
            }
        }
        
        self.config = self.defaults
        if config_path and os.path.exists(config_path):
            self.load_from_json(config_path)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            custom_config = json.load(f)
            self._update_recursive(self.config, custom_config)
            print(f"Loaded config from {path}")

    def _update_recursive(self, base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                self._update_recursive(base[k], v)
            else:
                base[k] = v

    def get(self, section, key=None):
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key, None)

    # Helper properties for easy access
    @property
    def model_params(self): return self.config['model']
    @property
    def moe_params(self): return self.config['moe']
    @property
    def loss_params(self): return self.config['loss']
    @property
    def data_params(self): return self.config['data']
    @property
    def train_params(self): return self.config['train']

# Singleton instance or factory can be used. 
# For now, we instantiate in main.
