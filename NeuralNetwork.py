from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, l1=2048, l2=1024, l3=512):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128*128, l1),
            nn.ReLU(),
            # nn.Linear(8192, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, 9)       # 9 categories for age
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits