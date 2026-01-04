
import toml
import json
import os
import math

latency_profile_path = 'profiles/latency_profile.json'
def read_toml_config(path_cf):
    with open(path_cf, 'r') as f:
        result = toml.load(f)

    def update(target, src):
        for k, v in src.items():
            if k not in target:
                target[k] = v

    if 'global' in result:
        glb = result['global']
        for k, v in result.items():
            update(v, glb)
    return result


# data = read_toml_config('data/test_config.toml')


def get_config(exp_id=None):
    if exp_id is None:
        exp_id = os.getenv('EXP_ID')
    if exp_id is None or exp_id not in data:
        all_cf = list(data.keys())
        raise RuntimeError(
            f'error, EXP_ID is not set or not belong to: {all_cf}')
    return exp_id, data[exp_id]


def get_latency_profile(exp_id=None):
    with open(latency_profile_path, 'r') as f:
        latency_profile = json.load(f)
    new_latency_profile = {}
    for k, v in latency_profile.items():
        new_latency_profile[k] = {
            int(batch_size): latency for batch_size, latency in v.items()}
    latency_profile = new_latency_profile
    return latency_profile


def get_slo_factor_to_batch_size_table(exp_id=None, batch_upper_bound=40, step_delta=0.95):
    cf = get_config(exp_id)[1]['latency_profile']
    with open(cf, 'r') as f:
        latency_profile = json.load(f)
    slo_factor_to_batch_size = {}

    base_latency = latency_profile["unet"]['1']  # 基准延迟（batch_size=1时的延迟）

    for slo_factor in [i/10 for i in range(10, 401)]:  # 1.0 到 10.0，步长0.1
        max_batch_size = 1

        while max_batch_size+1 in latency_profile["unet"] and \
                latency_profile["unet"][max_batch_size+1] < slo_factor * base_latency * step_delta:
            max_batch_size += 1
            if max_batch_size >= batch_upper_bound:
                break

        slo_factor_to_batch_size[int(slo_factor*10)] = max_batch_size
    return slo_factor_to_batch_size


def get_ref_times(exp_id=None):
    # 用于统一不同测试下 输入数据的一致性(不同数据latency的测试数据不同，slo factor需要作用于相同的基准数据，保证输入SLO设置是相同的)
    return get_config(exp_id)[1]['ref_time_unit']


def get_item(key, default_value, exp_id=None):
    return get_config(exp_id)[1].pop(key, default_value)


class ConfigParser:
    def __init__(self, exp_id=None, batch_upper_bound=40, step_delta=1):
        self.latency_profile = get_latency_profile()

        self.slo_factor_to_batch_size = {}
        self.slo_factor_to_batch_size_pre_stage = {}
        self.slo_factor_to_batch_size_post_stage = {}

        # 基准延迟（batch_size=1时的延迟）
        base_latency = self.latency_profile["unet"][1]

        time_per_batch = self.latency_profile["unet"][batch_upper_bound] / batch_upper_bound
        for slo_factor in [i/10 for i in range(10, 401)]:  # 1.0 到 10.0，步长0.1
            max_batch_size = 0

            while True:
                if max_batch_size+1 in self.latency_profile["unet"]:
                    next_time = self.latency_profile["unet"][
                        max_batch_size + 1]
                else:
                    next_time = time_per_batch * (max_batch_size + 1)
                if next_time < slo_factor * base_latency * step_delta:
                    max_batch_size += 1
                else:
                    break

            self.slo_factor_to_batch_size[int(slo_factor*10)] = max_batch_size

        # 基准延迟（batch_size=1时的延迟）
        base_latency = self.latency_profile["clip"][1]

        time_per_batch = self.latency_profile["clip"][
            batch_upper_bound] / batch_upper_bound
        for slo_factor in [i/10 for i in range(10, 401)]:  # 1.0 到 10.0，步长0.1
            max_batch_size = 0

            while True:
                if max_batch_size+1 in self.latency_profile["clip"]:
                    next_time = self.latency_profile["clip"][max_batch_size + 1]
                else:
                    next_time = time_per_batch * (max_batch_size + 1)
                if next_time < slo_factor * base_latency * step_delta:
                    max_batch_size += 1
                else:
                    break

            self.slo_factor_to_batch_size_pre_stage[int(
                slo_factor*10)] = max_batch_size

        # 基准延迟（batch_size=1时的延迟）
        base_latency = self.latency_profile["vae"][1] + \
            self.latency_profile["safety"][1]
        time_per_batch = self.latency_profile["vae"][
            batch_upper_bound] / batch_upper_bound + self.latency_profile["safety"][batch_upper_bound] / batch_upper_bound
        for slo_factor in [i/10 for i in range(10, 401)]:  # 1.0 到 10.0，步长0.1
            max_batch_size = 0

            while True:
                if max_batch_size+1 in self.latency_profile["vae"]:
                    next_time = self.latency_profile["vae"][max_batch_size+1] + self.latency_profile["safety"][max_batch_size+1]
                else:
                    next_time = time_per_batch * (max_batch_size + 1)
                if next_time < slo_factor * base_latency * step_delta:
                    max_batch_size += 1
                else:
                    break

            self.slo_factor_to_batch_size_post_stage[int(
                slo_factor*10)] = max_batch_size

        # print(self.data.keys())
        # self.reference_time = self.data['ref_time_unit']



    def get_latency_profile(self):
        return self.latency_profile

    def get_max_batch_size_post_stage(self, slo_factor10):
        return self.slo_factor_to_batch_size_post_stage[slo_factor10]

    def get_max_batch_size(self, slo_factor10):
        if slo_factor10 >= 400:
            return int(self.slo_factor_to_batch_size[400] * slo_factor10 / 400)
        else:
            return self.slo_factor_to_batch_size[slo_factor10]

    def get_slo_factor_to_batch_size(self):
        return self.slo_factor_to_batch_size


if __name__ == "__main__":
    print('config: ', get_config())
