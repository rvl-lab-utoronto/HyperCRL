import fire
import hypercrl

import sys


if __name__ == "__main__":
    fire.Fire({ 
        'coreset': hypercrl.coreset,
        'pnn':  hypercrl.pnn,
        'finetune':  hypercrl.finetune,
        'single':  hypercrl.single,
        "ewc":  hypercrl.ewc,
        "si":  hypercrl.si,
        "multitask":  hypercrl.multitask,
        'hnet':  hypercrl.hnet,
        'chunked_hnet':  hypercrl.chunked_hnet,
        'hnet_mt': hypercrl.hnet_mt,
        'hnet_replay': hypercrl.hnet_replay
    })
