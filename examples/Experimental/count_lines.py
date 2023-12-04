def count_lines(toml_path):
    with open(toml_path, "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if x and not x.startswith("#")]
        print(f"{toml_path} has {len(lines)} lines")
        return len(lines)