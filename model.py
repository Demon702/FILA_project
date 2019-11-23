import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy 
class Model(nn.Module):
    def __init__(self , size):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(2,1, 2)
        self.cnn2 = nn.Conv2d(2,1, 2)
        self.cnn3 = nn.Conv2d(2,1, 2)
        self.cnn4 = nn.Conv2d(2,1, 2)
        # self.cnn2 = nn.Conv2d(4,8, 2)
        self.size = size
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4 * (self.size -2) * (self.size -2) , 16 * (self.size -2) * (self.size -2))
        self.fc2 = nn.Linear(16 * (self.size -2) * (self.size -2) , self.size * (self.size - 1) * 2)
        self.sigmoid = nn.Sigmoid()
        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        # model_ft = models.resnet34  (pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).    
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        # model_ft.load_state_dict(torch.load("model34"))
        # self.feature_extractor_part1 = model_ft
        # num_ftrs = self.feature_extractor_part1.fc.out_features
        # # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).    

        # # self.feature_extractor_part1 = nn.Sequential(
        # #     nn.Conv2d(3, 20, kernel_size=5),
        # #     nn.ReLU(),
        # #     nn.MaxPool2d(2, stride=2),
        # #     nn.Conv2d(20, 50, kernel_size=5),
        # #     nn.ReLU(),
        # #     nn.MaxPool2d(2, stride=2)
        # # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(num_ftrs, self.L),
        #     nn.Sigmoid(),
        # )

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh()
        # )

        # self.gate = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Sigmoid()
        # )

        # self.weight = nn.Linear(self.D , self.K)

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.L*self.K, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # x = x.squeeze(0)
        input = input.float()
        # print(input.shape)
        rows = input[0]
        rows = rows.unsqueeze(0).unsqueeze(0)
        columns = input[1]
        columns = torch.transpose(columns , 0, 1)
        columns = columns.unsqueeze(0).unsqueeze(0)
        # print(columns.shape)
        # print(torch.cat((rows[:,:, :-1, :] , columns[:,:,:,:-1]) , 0).shape)
        feature1 = self.sigmoid(self.cnn1(torch.cat((rows[:,:, :-1, :] , columns[:,:,:,:-1]) , 1)))
        feature2 = self.sigmoid(self.cnn2(torch.cat((rows[:,:, :-1, :] , columns[:,:,:,1:]) , 1)))
        feature3 = self.sigmoid(self.cnn3(torch.cat((rows[:,:, 1:, :] , columns[:,:,:,:-1]) , 1)))
        feature4 = self.sigmoid(self.cnn4(torch.cat((rows[:,:, 1:, :] , columns[:,:,:,1:]) , 1)))
        
        features = torch.cat((feature1, feature2 , feature3 , feature4) , 0)
        all_features = features.view(-1 , 4 * (self.size -2) * (self.size -2))

        # print(H.shape)
        # H = H.view(-1, 50 * 53 * 53)
        # H = self.feature_extractor_part2(H)  # NxL


        all_features = self.sigmoid(self.fc1(all_features))
        Q_values = self.fc2(all_features)
        return Q_values.squeeze(0)

    # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

    #     return error, Y_hat

