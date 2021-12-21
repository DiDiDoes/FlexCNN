from nas_201_api import NASBench201API as API


if __name__ == "__main__":
    api = API('/nfs/data/chengdicao/data/NAS-Bench-201/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    #api = API('/nfs/data/chengdicao/data/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth', verbose=True)

    for i in range(5):
        config = api.get_net_config(i, 'cifar10')
        print(config)

