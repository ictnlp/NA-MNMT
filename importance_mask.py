import numpy as np 
import pickle
import torch

# lang1,lang2,lang3,lang4 are languages pairs of your model,
# such as "en-zh,zh-en,en-ar,ar-en"

lang1 = open('importance_value/lang1', 'rb')
lang2 = open('importance_value/lang2', 'rb')
lang3 = open('importance_value/lang3', 'rb')
lang4 = open('importance_value/lang4', 'rb')
taylor_1 = pickle.load(lang1)
taylor_2 = pickle.load(lang2)
taylor_3 = pickle.load(lang3)
taylor_4 = pickle.load(lang4)
taylor = [taylor_1, taylor_2, taylor_3, taylor_4]
lang_str = ['lang1', 'lang2', 'lang3', 'lang4']
#find the input value of the parameters
n_dic = {}
n = 4 # the number of languages pairs

number = {'1':'0', '2':'1', '3':'2', '4':'3', '5':'4'}

for name in taylor_1.keys():
    if name == 'decoder.embed_out':
        n_dic[name] = 'decoder.layers.5.final_layer_norm'
    else:
        tmp = name.split('.')
        if tmp[-1] == 'in_v' or tmp[-1] == 'in_q' or tmp[-1] == 'in_k':
            if tmp[-1] == 'in_v':
                if tmp[2] == '0':
                    n_dic[name] = tmp[0] + '.embed_tokens'
                else:
                    n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + number[tmp[2]] + '.' + 'final_layer_norm'
            else:
                if tmp[-2] == 'encoder_attn':
                    n_dic[name] = 'encoder.layers.5.final_layer_norm'
                else:
                    if tmp[2] == '0':
                        n_dic[name] = tmp[0] + '.embed_tokens'
                    else:
                        n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + number[tmp[2]] + '.' + 'final_layer_norm'
        elif tmp[-1] == 'out_proj':
            n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.' + tmp[3] + '.' + 'in_v'
        elif tmp[-1] == 'self_attn_layer_norm':
            n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.' + 'self_attn.out_proj'
        elif tmp[-1] == 'encoder_attn_layer_norm':
            n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.' + 'encoder_attn.out_proj'
        elif tmp[-1] == 'fc1':
            if tmp[0] == 'encoder':
                n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.' +'self_attn_layer_norm'
            else:
                n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.' +'encoder_attn_layer_norm'
        elif tmp[-1] == 'fc2':
            n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.fc1'
        elif tmp[-1] == 'final_layer_norm':
            n_dic[name] = tmp[0] + '.' + tmp[1] + '.' + tmp[2] + '.fc2'
        else:
            if name == 'encoder.embed_tokens' or name == 'decoder.embed_tokens':
                print(name)
            else:
                print('No right module')
                exit(-1)

src_embed = 15896
tgt_embed = 16008
ratio = 0.7
partion = 0.9

def get_fixed_index(input, mean_input):
    col = []
    index = []
    for i in range(input[0].size(0)):
        flag = 0
        for lang in input:
            if lang[i] < mean_input:
                continue
            flag = flag + 1
        if flag == n: 
            index.append(i)
    return torch.tensor(index)

def get_index(sorted_index, parts, partion):
    for i in sorted_index:
        if parts[i] <= partion:
            return i

def get_mask(input, fixed_index, weight, row=False):
    masks = []
    index_lang = [[], [], [], [], [], [], [], []]
    
    #针对每个神经元，跳过G的，将最大的分配给其相关的
    for i in range(input[0].size(0)):
        if i in fixed_index:
            continue
        maxvalue = max([float(lang[i]) for lang in input])
        for j in range(len(input)):
            if input[j][i] > maxvalue*partion:
                index_lang[j].append(i)

    for i in index_lang:
        i = torch.tensor(i).cpu().long()
        i = torch.cat( (i, fixed_index.cpu().long()) ,dim = 0)#增加G参数到训练中
        masks.append(i)
    return masks, index_lang

def get_neuron_mask(input, num):
    sum_input = sum(input)
    fixed_index = torch.topk(sum_input, int(input[0].size(0) * ratio), dim=0)[-1]
    print(fixed_index.size(0)/input[0].size(0))

    matrix = torch.ones(input[0].size(0)).cpu()
    masks, index_lang = get_mask(input, fixed_index, matrix)

    matrix.scatter_(0, masks[num].long(), 0.)
    print(matrix.sum() / input[0].size(0))
    return matrix.cpu()

def get_weight_mask(input, output, num, norm=False):#一旦上一层是norm，那么只根据输出mask

    sum_input = sum(input)#512, input[5 x 512]
    fixed_index = torch.topk(sum_input, int(input[0].size(0) * ratio), dim=0)[-1]
    print(fixed_index.size(0)/input[0].size(0))

    weight = torch.ones(output[0].size(0), input[0].size(0)).cpu()
    col_masks, index_lang = get_mask(input, fixed_index, weight)
    #print(len(index_lang[0]), len(index_lang[1]), len(index_lang[2]), len(index_lang[3]), len(index_lang[4]), input[0].size(0))
    print(len(index_lang[0]), len(index_lang[1]))
    index_lang1 = index_lang
    
    sum_output = sum(output)
    fixed_index = torch.topk(sum_output, int(output[0].size(0) * ratio), dim=0)[-1]
    print(fixed_index.size(0)/output[0].size(0))
    row_masks, index_lang = get_mask(output, fixed_index, weight, row=True)


    weight = torch.ones(output[0].size(0), input[0].size(0)).cpu()

    if index_lang1[num] != []:
            weight.scatter_(-1, col_masks[num].long(), 0.)
    if index_lang[num] != []:
            weight.scatter_(0, row_masks[num].long(), 0.)
    #print(weight)
    #print((torch.sum(weight)/ weight.size(0) / weight.size(1)).float())
    #print(torch.sum(weight), weight.size(0), weight.size(1))
    return weight.cpu()
	#返回一个字典，那么在模型中读取时就是字典套字典，先模块后语种

def get_bias_mask(output, num):
    weight = torch.ones(output[0].size(0)).cpu()
    
    sum_output = sum(output)
    fixed_index = torch.topk(sum_output, int(output[0].size(0) * ratio), dim=0)[-1]
    col, index_lang = get_mask(output, fixed_index, weight)
    weights = []

    weight = torch.ones(output[0].size(0)).cpu()
    if index_lang[num] != []:
        weight.scatter_(0, torch.tensor(index_lang[num]).cpu(), 0.)
    return weight.cpu()

def get_output(taylor, name):
	out = []
	for t in taylor:
		out.append(t[name].float())
	return out

def get_input(taylor, n_dic, name):
	out = []
	for t in taylor:
		out.append(t[name])
	return out

def final(num):
    res = {}
    for name in taylor_1.keys():
        tmp = name.split('.')
        if name == 'encoder.embed_tokens':
            mask = torch.ones(src_embed, taylor[name].size(0)).cpu()
            col = torch.topk(taylor[name], int(taylor[name].size(0) * ratio), dim=0, largest=False)[-1]
            col_mask = torch.stack([col] * mask.size(0), dim=0)
            mask.scatter_(-1, col_mask, 0.)
            res[name+'.weight'] = mask
        elif name == 'decoder.embed_tokens':
            mask = torch.ones(tgt_embed, taylor[name].size(0)).cpu()
            col = torch.topk(taylor[name], int(taylor[name].size(0) * ratio), dim=0, largest=False)[-1]
            col_mask = torch.stack([col] * mask.size(0), dim=0)
            mask.scatter_(-1, col_mask, 0.)
            res[name+'.weight'] = mask

        elif name == 'decoder.embed_out':
            output = get_output(taylor, name)
            input = get_input(taylor, n_dic, name)
            norm = (n_dic[name].split('.')[-1].split('_')[-1] == 'norm')
            res[name] = get_neuron_mask(output, num)
        elif tmp[-1] == 'in_q':
            n_tmp = ''
            for i in range(len(tmp) - 1):
                n_tmp += tmp[i]
                n_tmp += '.'
            q_output = get_output(taylor, n_tmp + 'in_q')
            if (n_dic[n_tmp + 'in_q'] == 'encoder.embed_tokens' or n_dic[n_tmp + 'in_q'] == 'decoder.embed_tokens' or n_dic[n_tmp + 'in_v'] == 'decoder.embed_tokens'):
                continue
            q_input = get_input(taylor, n_dic, n_tmp + 'in_q')
            q_norm = (n_dic[name].split('.')[-1].split('_')[-1] == 'norm')
            q_matrix = get_neuron_mask(q_output, num)
            
            k_output = get_output(taylor, n_tmp + 'in_k')
            k_input = get_input(taylor, n_dic, n_tmp + 'in_k')
            k_norm = (n_dic[name].split('.')[-1].split('_')[-1] == 'norm')
            k_matrix = get_neuron_mask(k_output, num)

            v_output = get_output(taylor, n_tmp + 'in_v')
            v_input = get_input(taylor, n_dic, n_tmp + 'in_v')
            v_norm = (n_dic[name].split('.')[-1].split('_')[-1] == 'norm')
            v_matrix = get_neuron_mask(v_output, num)
            
            res[n_tmp+ 'in_q'] = q_matrix
            res[n_tmp+ 'in_k'] = k_matrix
            res[n_tmp+ 'in_v'] = v_matrix
        elif tmp[-1] == 'in_k' or tmp[-1] == 'in_v':
            continue
        else:
            #print(tmp)
            ttmp = tmp[-1].split('_')
            if ttmp[-1] != 'norm':
                output = get_output(taylor, name)
                input = get_input(taylor, n_dic, name)
                norm = (n_dic[name].split('.')[-1].split('_')[-1] == 'norm')
                res[name] = get_neuron_mask(output, num)
    return res


ress={}
for i in range(0, n):
    res = final(i)
    ress[lang_str[i]] = res
g = open('mask/taylor_mask', 'wb')
pickle.dump(ress, g)
