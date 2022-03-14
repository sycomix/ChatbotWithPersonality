
def get_model_params():
    params = {}

    # Encoder/Decoder similar params
    params["hidden_units_num"] = 512
    params["deep_layers_num"] = 2

    params["enable_residual_wrapper"] = True
    params["enable_dropout_wrapper"] = True

    params["input_keep_prob"] = 1.0 - 0.0
    params["output_keep_prob"] = 1.0 - 0.0
    params["embedding_file"] = "./embeddings/glove.6B.300d.txt"
    params["vocabulary_size"] = 40000
    params["embedding_size"] = 300
    params["batch_size"] = 30

    # Decoder additional params
    params["attention_method"] = "luong"
    params["beam_width"] = 30

    params["learning_rate"] = 0.002
    params["optimizer_type"] = "adam"
    params["gradient_clipping_norm"] = 1.0
    params["global_step"] = 1

    params["sentence_max_len"] = 20
    params["max_decode_step"] = 20

    return params