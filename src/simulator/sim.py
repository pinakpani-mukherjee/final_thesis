from mesh_objects import Mesh
import csv
import torch
from src.models import ASTModel
from torch.cuda.amp import autocast
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


# create mesh
# choose a point in mesh to be source
# calculate signal and signal strength in the surrounding points


class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


input_tdim = 1024
checkpoint_path = './../../pretrained_models/audio_mdl.pth'
ast_mdl = ASTModelVis(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
print(f'[*INFO] load checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
audio_model.load_state_dict(checkpoint)
audio_model = audio_model.to(torch.device("cuda:0"))
audio_model.eval()
label_csv = './../../egs/audioset/data/class_labels_indices.csv'
labels = load_label(label_csv)

x_length = 30
y_length = 30
src_1_x = 15
src_1_y = 10

src_2_x = 20
src_2_y = 5


def run_simulation():
    simulated_mesh = Mesh(x_length, y_length, x_tick=1, y_tick=1)

    simulated_mesh.set_source(audio_file_path="./../../sample_audios/simulation_audio/angry-dog.wav", x=src_1_x,
                              y=src_1_y, initial_spl_at_src=110)
    simulated_mesh.set_source(audio_file_path="./../../sample_audios/simulation_audio/angry-crowd.wav", x=src_2_x,
                              y=src_2_y, initial_spl_at_src=110)
    simulated_mesh.create_spls()

    for i in range(0, x_length):
        for j in range(0, y_length):
            with torch.no_grad():
                feats_data = simulated_mesh.mesh[i][j].signal_features_data.expand(1, input_tdim,
                                                                                   128)  # reshape the feature
                feats_data = feats_data.to(torch.device("cuda:0"))
                with autocast():
                    output = audio_model.forward(feats_data)
                    output = torch.sigmoid(output)
                    result_output = output.data.cpu().numpy()[0]
                    sorted_indexes = np.argsort(result_output)[::-1]
                    return_dict = OrderedDict()
                    for k in range(10):
                        return_dict[f"{np.array(labels)[sorted_indexes[k]]}"] = result_output[sorted_indexes[k]] * \
                                                                                simulated_mesh.mesh[i][j].spl
                    simulated_mesh.mesh[i][j].output_status = return_dict

    src_dict = simulated_mesh.mesh[int((src_1_x + src_2_x) / 2)][int((src_1_y + src_2_y) / 2)].output_status
    for output_name in list(src_dict.keys()):
        with open(f'./output_values/{output_name}.txt', 'w') as f:
            for i in range(0, x_length):
                for j in range(0, y_length):
                    f.write(f"{i} {j} {simulated_mesh.mesh[i][j].output_status.get(f'{output_name}', 0)}\n")


print("finished")

if __name__ == "__main__":
    run_simulation()
