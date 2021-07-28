
class Hparams():
    @staticmethod
    def add_hnet_hparams(hparams):
        # Hypernetwork
        if hparams.h_dims == [32, 32]:
            hparams.hnet_arch = [16, 16]
        elif hparams.h_dims == [200, 200]:
            hparams.hnet_arch = [50, 50]
        elif hparams.h_dims == [256, 256]:
            hparams.hnet_arch = [128, 128]
        elif hparams.h_dims == [200, 200, 200, 200]:
            hparams.hnet_arch = [256, 256]
        elif hparams.h_dims == [200, 200, 200]:
            hparams.hnet_arch = [100, 100]
        elif hparams.h_dims == [400, 400, 400]:
            hparams.hnet_arch = [100, 100]
        elif hparams.h_dims == [100, 100]:
            hparams.hnet_arch = [40, 40]

        if hparams.env == "door":
            hparams.hnet_act = "elu"
        elif hparams.env == "door_pose":
            hparams.hnet_act = "relu"
        elif hparams.env == "door_pose_kuka":
            hparams.hnet_act = "relu"
        elif hparams.env == "pusher":
            hparams.hnet_act = "elu"
        else:
            hparams.hnet_act = 'relu'

        # Embedding
        hparams.emb_size = 10
        # Initialization
        hparams.use_hyperfan_init = False
        hparams.hnet_init = "xavier" # or "normal"
        hparams.std_normal_init = 0.02
        hparams.std_normal_temb = 1 # std when initializing task embedding

        # Training param
        hparams.lr_hyper = 0.0001
        hparams.grad_max_norm = 5
    
        if hparams.env == "door_pose" or hparams.env == "door_pose_kuka" or hparams.env == "pusher_slide":
            hparams.beta = 0.5
        else:
            hparams.beta = 0.05

        hparams.no_look_ahead = False # False=use two step optimization
        hparams.plastic_prev_tembs = False # Allow adaptation of past task embeddings
        hparams.backprop_dt = False #Allow backpropagation through delta theta in the regularizer
        hparams.use_sgd_change = False # Approximate change with in delta theta with SGD
        hparams.ewc_weight_importance = False # Use fisher matrix to regularize
                                            # model weights generated from hnet
        hparams.n_fisher = -1 # Number of training samples to be used for the ' +
                            # 'estimation of the diagonal Fisher elements. If ' +
                            # "-1", all training samples are us

        hparams.si_eps = 1e-3
        hparams.mlp_var_minmax = True

        return hparams
    
    @staticmethod
    def add_chunked_hnet_hparams(hparams):
        # Hypernetwork 
        if hparams.h_dims == [256, 256]:
            hparams.hnet_arch = [5, 5]
            hparams.chunk_dim = 12000 # Chunk size (output dim of hnet)
            hparams.cemb_size = 40
        elif hparams.h_dims == [200, 200, 200, 200]:
            hparams.hnet_arch = [25, 30]
            hparams.chunk_dim = 4000
            hparams.cemb_size = 20
        elif hparams.h_dims == [200, 200]:
            hparams.hnet_arch = [20, 20]
            hparams.chunk_dim = 2000
            hparams.cemb_size = 20
        hparams.hnet_act = 'relu'

        # Embedding
        hparams.emb_size = 10
        # Initialization
        hparams.use_hyperfan_init = False
        hparams.hnet_init = "xavier" # or "normal"
        hparams.std_normal_init = 0.02
        hparams.std_normal_temb = 1 # std when initializing task embedding
        hparams.std_normal_cemb = 1

        # Training param
        hparams.lr_hyper = 0.0001
        hparams.grad_max_norm = 5
        hparams.beta = 0.005

        hparams.no_look_ahead = False # False=use two step optimization
        hparams.plastic_prev_tembs = True # Allow adaptation of past task embeddings
        hparams.backprop_dt = False #Allow backpropagation through delta theta in the regularizer
        hparams.use_sgd_change = False # Approximate change with in delta theta with SGD
        hparams.ewc_weight_importance = False # Use fisher matrix to regularize
                                            # model weights generated from hnet
        hparams.n_fisher = -1 # Number of training samples to be used for the ' +
                            # 'estimation of the diagonal Fisher elements. If ' +
                            # "-1", all training samples are us
        
        return hparams

def HP(env, seed=None, save_folder='./runs/lqr'):
    hparams = Hparams()
    hparams.seed = seed if seed is not None else 2020
    hparams.save_folder = save_folder if save_folder is not None else './runs/lqr'
    hparams.resume = False

    # Common train setting
    hparams.num_ds_worker = 0
    hparams.print_train_every = 1000

    # common RL setting
    hparams.env = env
    hparams.gt_dynamic = False
    hparams.gpuid = "cuda:0"
    
    if env == "lqr":
        return default_arg_2d_car(hparams)
    elif env == "lqr10":
        return default_arg_10d_car(hparams)
    elif env.startswith("hopper"):
        return default_arg_hopper(hparams)
    elif env == "humanoid":
        return default_arg_humanoid(hparams)
    elif env.startswith("half_cheetah"):
        return default_arg_half_cheetah(hparams)
    elif env.startswith("inverted_pendulum"):
        return default_arg_inverted_pendulum(hparams)
    elif env.startswith("pendulum"):
        return default_arg_pendulum(hparams)
    elif env == "cartpole":
        return default_arg_cartpole(hparams)
    elif env == "cartpole_bin":
        return default_arg_cartpole_bin(hparams)
    elif env == "metaworld10":
        return default_arg_metaworld10(hparams)
    elif env == "reacher":
        return default_arg_reacher(hparams)
    elif env == "pusher":
        return default_arg_pusher(hparams)
    elif env == "door":
        return default_arg_door(hparams)
    elif env == "door_pose":
        return default_arg_door_pose(hparams)
    elif env == "door_pose_kuka":
        return default_arg_door_pose(hparams)
    elif env == "pusher_rot":
        return default_arg_pusher_rot(hparams)
    elif env == "pusher_slide":
        return default_arg_pusher_slide(hparams)

def default_arg_metaworld10(hparams):
    hparams.state_dim = 9
    hparams.control_dim = 4
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 30000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1500

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.0001
    hparams.train_dynamic_iters = 10000
    hparams.eval_every = 5000

    # RL Eval setting
    hparams.eval_env_run_every = 1500
    hparams.run_eval_env_eps = 5

    # Size of inducing points
    hparams.M = 400

    # RL Planning
    hparams.control = "mpc-pddm"
    hparams.horizon = 7
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 2000 # Number of traj to sample
    hparams.num_cem_elites = 10

    # PDDM
    hparams.pddm_beta = 0.8
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams

def default_arg_humanoid(hparams):
    hparams.state_dim = 376
    hparams.control_dim = 17
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.max_iteration = 40001
    hparams.init_rand_steps = 1000
    hparams.dynamics_update_every = 10000

    # Common Dynamics Model
    hparams.dnn_out = "state" # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]
    hparams.out_var = False

    hparams.lr = 0.0001
    hparams.lr_steps = [8500]
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 10000
    hparams.eval_every = 5000

    # Size of inducing points
    hparams.M = 50

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 10 # Number of search steps
    hparams.n_sim_particles = 100 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams

def default_arg_hopper(hparams):
    hparams.state_dim = 12
    hparams.control_dim = 3
    hparams.out_dim = hparams.state_dim

     # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 100000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1000

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]
    hparams.out_var = True

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.000075
    hparams.train_dynamic_iters = 2000
    hparams.eval_every = 2000

    # RL Eval setting
    hparams.eval_env_run_every = 5000
    hparams.run_eval_env_eps = 4

    # Size of inducing points
    hparams.M = 1000

    # RL Planning
    hparams.control = "mpc-pddm"
    hparams.horizon = 7
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 2500 # Number of traj to sample
    hparams.num_cem_elites = 50

    # PDDM
    hparams.pddm_beta = 0.7
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams

def default_arg_pendulum(hparams):
    hparams.state_dim = 3
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 400
    hparams.max_iteration = 10000
    hparams.dynamics_update_every = 400

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [32, 32]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 20
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 1000
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 50

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 15
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 10 # Number of search steps
    hparams.n_sim_particles = 100 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams

def default_arg_inverted_pendulum(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 3
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 40000
    hparams.dynamics_update_every = 1000
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0.0001
    hparams.train_dynamic_iters = 2000
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 2000

    # Size of inducing points
    hparams.M = 400

    # RL Eval setting
    hparams.eval_env_run_every = 4000
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 1000 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 10

    return hparams

def default_arg_half_cheetah(hparams):
    hparams.state_dim = 18
    hparams.control_dim = 6
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 100000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1000

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]
    hparams.out_var = True

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.000075
    hparams.train_dynamic_iters = 2000
    hparams.eval_every = 2000

    # RL Eval setting
    hparams.eval_env_run_every = 5000
    hparams.run_eval_env_eps = 1

    # Size of inducing points
    hparams.M = 1000

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 30
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 500 # Number of traj to sample
    hparams.num_cem_elites = 50

    # PDDM
    hparams.pddm_beta = 0.7
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams

def default_arg_cartpole(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 10
    hparams.init_rand_steps = 400
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 30

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 400 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    hparams.mag_noise = 1

    return hparams

def default_arg_cartpole_bin(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 3
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 5000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 200

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 400 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    return hparams

def default_arg_2d_car(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 2
    hparams.max_iteration = 4000
    hparams.init_rand_steps = 2000
    hparams.dynamics_update_every = 200

    # Common Dynamics Model
    hparams.dnn_out = "state" # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [32, 32]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = [4500]
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 5000
    hparams.eval_every = 2500

    # Size of inducing points
    hparams.M = 50

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 200
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # CEM
    hparams.n_sim_steps = 10 # Number of search steps
    hparams.n_sim_particles = 100 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams

def default_arg_10d_car(hparams):
    hparams.state_dim = 20
    hparams.control_dim = 10
    hparams.out_dim = hparams.state_dim
    hparams.rand_aggregate_seed = 2020

    # Tasks
    hparams.num_tasks = 4
    hparams.max_iteration = 1
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 400
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "diff"
    hparams.normalize_xu = False
    hparams.h_dims = [32, 32]

    hparams.lr = 0.0001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 50000
    hparams.eval_every = 2500

    # Size of inducing points
    hparams.M = 50

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 30
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # RL Eval setting
    hparams.eval_env_run_every = 400
    hparams.run_eval_env_eps = 1

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 10000 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams

def default_arg_reacher(hparams):
    hparams.state_dim = 11
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 10
    hparams.init_rand_steps = 200
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 50
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 150
    hparams.print_train_every = 150
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 150

    # Size of inducing points
    hparams.M = 30

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 400 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 10
    hparams.mag_noise = 1

    return hparams

def default_arg_pusher(hparams):
    hparams.state_dim = 10
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 500 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams

def default_arg_pusher_rot(hparams):
    hparams.state_dim = 20
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 500 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams

def default_arg_pusher_slide(hparams):
    hparams.state_dim = 18
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 300
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 150
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 500 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams

def default_arg_door(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 3
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 1
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 500 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams

def default_arg_door_pose(hparams):
    hparams.state_dim = 26
    hparams.control_dim = 7
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 60000
    hparams.dynamics_update_every = 200
    hparams.out_var = True

    # Common Dynamics Model
    hparams.dnn_out = "diff" # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0.00001
    hparams.train_dynamic_iters = 200
    hparams.print_train_every = 200
    hparams.gpuid = 'cuda:0'
    hparams.eval_every = 200

    # Size of inducing points
    hparams.M = 600

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 10
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5 # Number of search steps
    hparams.n_sim_particles = 2000 # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 0.5

    return hparams