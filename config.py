class Config(object):

    def __init__(self):
        
        
        
        self.cnn_architecture = 'resnet50'
        self.rnn_architecture = 'lstm'
        
        self.pretrain = True
        
        self.own_model = False       
       
        self.data_dir = './data/'
        
        self.lang = 'en'
        
        self.show_val = False
        self.num_epochs = 30
        self.batch_size = 32
        
        self.translate = False
        self.mode = 'train'
        
        self.show_size = 50       
       
        
        self.embed_drop_rate = 0.5
        
        self.rnn_drop_rate = 0.3
        
        self.hacked_val = False
        
        self.momentum = 0.9
        
        self.decoder_type = 'beam'
        self.embed_size = 512
        
        self.rnn_num_layers = 2
        self.hidden_size = 512        
        self.mod_lr = 0.8
        self.decode_layer = 1024        
        self.threshold = 1
        self.train_rnn = True
        self.train_cnn = False
       
        self.log_step = 10
        self.save_step = 500        
        self.image_size = 256        
        self.crop_size = 224        
        
        self.sample_batch_size = 32
        self.top_k = 5
        self.alpha = 0
        self.max_caption_length = 50
        self.loader = True      
        
        self.train_image_dir = './data/train_set/'
        self.val_image_dir = './data/val_set/'
        
        self.res_train = './data/resized_train_set/'
        self.res_val = './data/resized_val_set/'
        
        self.test_set = './data/test_set/'
        self.en_test_cap = './data/testendata.csv'
        self.de_test_cap = './data/testdedata.csv'
        
        self.en_caption_dir = './data/trainendata.csv'
        self.de_caption_dir = './data/traindedata.csv'
        
        
        self.en_val_caption_dir = './data/envalsingle.csv'
        self.de_val_caption_dir = './data/devalsingle.csv'
        
        self.en_vocab_dir = './data/vocab_en.pkl'
        
        self.de_vocab_dir = './data/vocab_de.pkl'
        
        self.model_dir = './models/'
        
        self.given_weights = False
        self.clip_val = 5
        self.attention = True
        self.attention_dim = 512
        
        if self.cnn_architecture == 'resnet50':
            self.encoded_image_size = 7
        else:
            self.encoded_image_size = 14
        self.encoder_dim = 0
        self.alpha_c = 1
        self.prob = 0
        self.num_workers = 2        
        self.sortData = True
