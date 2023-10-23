
def get_model_params():
    return {
        "hidden_units_num": 512,
        "deep_layers_num": 2,
        "enable_residual_wrapper": True,
        "enable_dropout_wrapper": True,
        "input_keep_prob": 1.0 - 0.0,
        "output_keep_prob": 1.0 - 0.0,
        "embedding_file": "./embeddings/glove.6B.300d.txt",
        "vocabulary_size": 40000,
        "embedding_size": 300,
        "batch_size": 30,
        "attention_method": "luong",
        "beam_width": 30,
        "learning_rate": 0.002,
        "optimizer_type": "adam",
        "gradient_clipping_norm": 1.0,
        "global_step": 1,
        "sentence_max_len": 20,
        "max_decode_step": 20,
    }