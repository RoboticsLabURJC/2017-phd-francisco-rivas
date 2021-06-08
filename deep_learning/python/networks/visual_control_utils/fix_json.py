def read_malformed_json(input_file, ):
    fixed_data = []
    # We process json
    data_parse = open(input_file).read().split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        fixed_data.append({"v": float(v), "w": float(w)})

    return fixed_data
