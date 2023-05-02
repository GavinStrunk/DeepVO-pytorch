import os

def get_dataset_dir_root():
    """Checks for the DATASET_DIR_ROOT environment variable

    This environment variable is assumed to be set to the root directory of all datasets.
    :return: absolute directory path
    :raises: EnvironmentError
    """
    dataset_dir = os.environ.get('DATASET_DIR_ROOT')
    if dataset_dir is None:
        raise EnvironmentError(
            "Could not find environment variable DATASET_DIR_ROOT. Set this variable to the root directory of the dataset folder")

    return dataset_dir

class Parameters():
    def __init__(self):
        self.n_processors = 64
        # Path
        self.dataset_dir = get_dataset_dir_root() + '/KITTI/odometry/dataset'

        self.train_video = ['00', '01', '02', '08', '09']
        self.valid_video = ['03', '04', '05', '06', '07', '10']
        # self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8

        # Data Preprocessing
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 1280  # original size is about 1226
        self.img_h = 384  # original size is about 370
        self.img_means = (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
        self.img_stds = (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
        self.minus_point_5 = True

        # self.seq_len = (5, 7)
        # self.sample_times = 3

        # Data info path
        # self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
        #     ''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1],
        #     self.sample_times)
        # self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
        #     ''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1],
        #     self.sample_times)

        # Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.clip = None
        self.batch_norm = True
        # Training
        self.epochs = 200
        self.batch_size = 64
        self.pin_mem = True
        self.optim = {'opt': 'Adagrad', 'lr': 0.001}
        # Choice:
        # {'opt': 'Adagrad', 'lr': 0.001}
        # {'opt': 'Adam'}
        # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

        # Pretrain, Resume training
        # self.pretrained_flownet = None
        # Choice:
        # None
        # './pretrained/flownets_bn_EPE2.459.pth.tar'
        # './pretrained/flownets_EPE1.951.pth.tar'
        # self.resume = True  # resume training
        # self.resume_t_or_v = '.train'
        # self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video),
        #                                                                                    ''.join(self.valid_video),
        #                                                                                    self.img_h, self.img_w,
        #                                                                                    self.seq_len[0],
        #                                                                                    self.seq_len[1],
        #                                                                                    self.batch_size,
        #                                                                                    self.rnn_hidden_size,
        #                                                                                    '_'.join(
        #                                                                                        [k + str(v) for k, v in
        #                                                                                         self.optim.items()]),
        #                                                                                    self.resume_t_or_v)
        # self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(
        #     ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0],
        #     self.seq_len[1], self.batch_size, self.rnn_hidden_size,
        #     '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        #
        # self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video),
        #                                                                             ''.join(self.valid_video),
        #                                                                             self.img_h, self.img_w,
        #                                                                             self.seq_len[0], self.seq_len[1],
        #                                                                             self.batch_size,
        #                                                                             self.rnn_hidden_size, '_'.join(
        #         [k + str(v) for k, v in self.optim.items()]))
        # self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video),
        #                                                                                  ''.join(self.valid_video),
        #                                                                                  self.img_h, self.img_w,
        #                                                                                  self.seq_len[0],
        #                                                                                  self.seq_len[1],
        #                                                                                  self.batch_size,
        #                                                                                  self.rnn_hidden_size, '_'.join(
        #         [k + str(v) for k, v in self.optim.items()]))
        # self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(
        #     ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0],
        #     self.seq_len[1], self.batch_size, self.rnn_hidden_size,
        #     '_'.join([k + str(v) for k, v in self.optim.items()]))

        # if not os.path.isdir(os.path.dirname(self.record_path)):
        #     os.makedirs(os.path.dirname(self.record_path))
        # if not os.path.isdir(os.path.dirname(self.save_model_path)):
        #     os.makedirs(os.path.dirname(self.save_model_path))
        # if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
        #     os.makedirs(os.path.dirname(self.save_optimzer_path))
        # if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
        #     os.makedirs(os.path.dirname(self.train_data_info_path))



par = Parameters()
