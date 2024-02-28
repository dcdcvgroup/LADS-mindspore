import yaml
import os
from mindformers import AutoTokenizer, AutoModel
import shutil


def generate_bert_config(bert_model_root: str = "./checkpoint_download/bert", bert_model: str = 'bert_base_uncased', batch_size: int = 16, token_max_len: int = 20, text_encoder_layer_num: int = 12):
    custom_bert_config_root=os.path.join(bert_model_root,bert_model+'_'+str(batch_size) +
                    '_'+str(token_max_len)+'_'+str(text_encoder_layer_num))
    custom_bert_config_path = os.path.join(custom_bert_config_root,bert_model+'.yaml')
    original_bert_config_path = os.path.join(
        bert_model_root, bert_model+'.yaml')

    # if path is not exist, download the config files and ckpt
    if not (os.path.exists(original_bert_config_path)):
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModel.from_pretrained(bert_model)

    
    if not os.path.exists(custom_bert_config_path):
        os.makedirs(custom_bert_config_root)

    with open(original_bert_config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # change configs
    config_dict['model']['model_config']['batch_size'] = batch_size
    config_dict['model']['model_config']['num_labels'] = 2
    config_dict['model']['model_config']["seq_length"] = token_max_len
    config_dict['model']['model_config']['num_layers'] = text_encoder_layer_num

    # write configs into yaml
    with open(custom_bert_config_path, 'w') as nf:
        yaml.dump(config_dict, nf)
    shutil.copy(os.path.join(bert_model_root,'vocab.txt'),custom_bert_config_root)

    # bert_model_config=MyBertConfig(batch_size,token_max_len,text_encoder_layer_num)
    # model=AutoModel.from_pretrained('bert_base_uncased')


if __name__ == '__main__':
    # you can add some options in the function
    generate_bert_config(batch_size=32,token_max_len=40,text_encoder_layer_num=6)
