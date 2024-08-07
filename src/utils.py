import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        total_loss += loss

    return total_loss / len(train_loader)

def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return total_loss / len(valid_loader), correct / len(valid_loader.dataset)


def chebyshev_polynomials(x, degree):
    # T_0(x) and T_1(x)
    if degree == 0:
        return np.ones_like(x)
    elif degree == 1:
        return x
    else:
        Tn_2 = np.ones_like(x)
        Tn_1 = x
        Tn = None
        for n in range(2, degree + 1):
            Tn = 2 * x * Tn_1 - Tn_2
            Tn_2, Tn_1 = Tn_1, Tn
        return Tn

def coeffs_plot(coeffs, layer_num=1, layer_size=28*28, epoch=0, node_num=None, save_dir="/content", save_img=False):
    plt.figure(figsize=(8, 4))
    # choose the input dim (which node)
    if node_num==None: node_num=layer_size//2
    all_coeffs = coeffs[node_num]
    for idx, some_coeffs in enumerate(all_coeffs):
    x_values = np.linspace(-1, 1, 400)
    y_values = np.zeros_like(x_values)
    for i, coeff in enumerate(some_coeffs):
        y_values += coeff * chebyshev_polynomials(x_values, i)
    plt.plot(x_values, y_values, label=str(idx))
    
    img_name = f"layer_{layer_num}_node_{node_num}_epoch_{(epoch):02d}"

    # Figure ylim is dataset specific
    if layer_num == 1:
        plt.ylim(-0.035, 0.035)
    elif layer_num == 2:
        plt.ylim(-0.25, 0.25)
    elif layer_num == 3:
        plt.ylim(-2.5, 2.5)
    plt.title(f"Chebyshev Coefficients' Representation {img_name}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    # plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    if save_img: fig.savefig(f"{save_dir}/layer_{layer_num}/{img_name}.png")
    return fig

def convert_scripted_to_original(scripted_model, new_model):
    # Copy parameters
    for name, param in scripted_model.named_parameters():
        original_param = dict(new_model.named_parameters())[name]
        original_param.data.copy_(param.data)
    # Copy buffers
    for name, buffer in scripted_model.named_buffers():
        original_buffer = dict(new_model.named_buffers())[name]
        original_buffer.data.copy_(buffer.data)
    return new_model
  
def extract_features(text, extractor, tokenizer):
    # Tokenize the text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Get the hidden states for each token
    with torch.no_grad():
        outputs = extractor(input_ids)
        hidden_states = outputs[2]
    # Concatenate the last 4 hidden states
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])
    # Calculate the mean of the last 4 hidden states
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))
    # Return the features as a tensor
    return torch.stack(features)

def get_ds_items(ds, extractor, tokenizer):
    texts, labels, features = list(), list(), list()
    for item in tqdm(list(ds)):
        text, label = item[1], item[0]-1 # start label from 0 instead of 1
        feature = torch.flatten(extract_features(text, extractor, tokenizer))
        texts.append(text)
        labels.append(label)
        # normalization
        min_val = torch.min(feature)
        max_val = torch.max(feature)
        norm_feature = (feature - min_val) / (max_val - min_val)
        feature = torch.add(torch.mul(norm_feature, 2), -1) #feature = norm_feature * 2 - 1
        features.append(feature)
    print(f"\nds text len: {len(texts)}")
    print(f"ds labels len: {len(labels)}\t, unique labels len: {len(set(labels))}\t, labels: {set(labels)}") # 1-World, 2-Sports, 3-Business, 4-Sci/Tech
    print(f"ds features len: {len(features)}\t, max: {np.max(features)}\t, min: {np.min(features)}\n")
    return texts, labels, features

def save_pkl(ds, dir, split):
    save_dir = os.path.join(dir, f"AG_NEWS_{split}.pkl")
    with open(save_dir, 'wb') as f:
        pkl.dump(ds, f)

def load_pkl(dir, split):
    load_dir = os.path.join(dir, f"AG_NEWS_{split}.pkl")
    with open(load_dir, 'rb') as f:
        ds = pkl.load(f)
    return ds