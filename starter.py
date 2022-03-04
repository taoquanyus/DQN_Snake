import torch

from agent import Agent

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    use_gpu = torch.cuda.is_available()

    # Decide which device we want to run on
    print(torch.cuda.get_device_name(0))

    agent = Agent()
    agent.train()