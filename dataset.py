import pickle
import torch




# train
loop = 8
sample = 5000

# valid / test
valid_size = 5000
test_size = 20000 

def get_dataset_size():
    return valid_size, test_size


def get_np_dataset():
    with open("data.p", 'rb') as f:
        mnist = pickle.load(f)

    Data = mnist["data"]
    Data["target"] = mnist["target"]
    Data = Data.sample(frac=1, random_state=1998, ignore_index=True)




    train_image_set = Data.iloc[:-1 * test_size,:-1]
    train_label_set = Data.iloc[:-1 * test_size,-1:]

    np_vaild_image_set = train_image_set.iloc[-1 * valid_size:,:].to_numpy().reshape(-1,28,28)
    np_vaild_label_set = train_label_set.iloc[-1 * valid_size:,:].astype(int).to_numpy()
    np_train_image_set = train_image_set.iloc[:-1 * valid_size,:].to_numpy().reshape(-1,28,28)
    np_train_label_set = train_label_set.iloc[:-1 * valid_size,:].astype(int).to_numpy()
    np_test_image_set = Data.iloc[-1 * test_size:,:-1].to_numpy().reshape(-1,28,28)
    np_test_label_set = Data.iloc[-1 * test_size:,-1:].astype(int).to_numpy()

    # train_image_set = torch.from_numpy(np_train_image_set).to(torch.float32)
    # train_label_set = torch.from_numpy(np_train_label_set).to(torch.long).squeeze(dim=1)
    # vaild_image_set = torch.from_numpy(np_vaild_image_set).to(torch.float32)
    # vaild_label_set = torch.from_numpy(np_vaild_label_set).to(torch.long).squeeze(dim=1)
    # test_image_set = torch.from_numpy(np_test_image_set).to(torch.float32)
    # test_label_set = torch.from_numpy(np_test_label_set).to(torch.long).squeeze(dim=1)

    rtn = [[np_train_image_set,np_train_label_set],
           [np_vaild_image_set,np_vaild_label_set],
           [np_test_image_set,np_test_label_set]]

    return rtn