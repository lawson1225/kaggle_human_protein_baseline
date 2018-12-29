class DefaultConfigs(object):
    train_data = "../Human_Protein_Atlas/input/train/" # where is your train data
    test_data = "../Human_Protein_Atlas/input/test/"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 40
    epochs = 50
    resume = True
    initial_checkpoint = '0'

config = DefaultConfigs()
