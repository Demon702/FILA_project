import numpy as np
import pickle
import pygame   
import sys, pdb
import argparse
from model import Model
import torch
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class State:
	def __init__(self, p1, p2 , size = 10):
		self.size = size
		self.total_squares = (size - 1)**2
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		self.p1 = p1
		self.p2 = p2
		self.isEnd = False
		self.boardHash = None
		# init p1 plays first
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)
		self.loss = nn.MSELoss()

	def getHash(self):
		string = str(self.board.reshape((size - 1) * size * 2))
		self.boardHash = int(string , 2)
		return self.boardHash

	def reset(self):
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)

	def available_actions(self):
		available_array = np.where(self.board == 0)
		available_array = np.array(np.transpose(available_array))
		available_actions = [int("".join(map(str , index)) , 2) for index in available_array]
		return available_actions

	def updatestate(self, action):
		string = bin(action)
		grid_index  = [[int(i)] for i in string]
		self.grid_index = 1
		self.getHash()

	def winner(self):
		if sum(sum(self.grid_winners) == 1) > self.grid_winners.size/2:
			self.isEnd = True
			return 1
		elif sum(sum(self.grid_winners) == -1) > self.grid_winners.size/2:
			self.isEnd = True
			return -1
		else:
			return 0

	def reward(self , action , player):
		action_string = bin(action)
		[x, y, z]  = [int(i) for i in string]
		if z == 0:
			if self.board[x-1 , y, 0] and self.board[y, x-1, 1] and self.board[y+1, x-1, 1]:
				self.grid_winners[x-1 , y] = player
				self.no_of_boards += 1
				return 1

			elif self.board[x+1 , y, 0] and self.board[y, x, 1] and self.board[y+1, x, 1]:
				self.grid_winners[x-1 , y] = player
				self.no_of_boards += 1
				return 1
		elif z == 1:
			if self.board[y+1, x, 0] and self.board[y, x, 0] and self.board[x+1, y, 1]:
				self.grid_winners[y , x] = player
				self.no_of_boards += 1
				return 1
			elif self.board[x-1 , y, 1] and self.board[y, x-1, 0] and self.board[y+1, x-1, 0]:
				self.grid_winners[y , x-1] = player
				self.no_of_boards += 1
				return 1
		return 0

	def play(self, rounds=10000):
		for i in range(rounds):
			if i % 1000 == 0:
				print("Rounds {}".format(i))
			while not self.isEnd:
				# Player 1
				reward = 1
				while reward == 1 and not self.isEnd:
					actions = self.available_actions
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					p1_action = self.p1.chooseAction(actions, self.board)
					if self.p1.random_action_taken:
						current_Q_value = self.next_state_Q_value[p1_action]
					else:
						current_Q_value = self.p1.current_Q_value
					reward = self.reward(p1_action , self.p1.symbol)
					if self.winner() == p1.symbol:
						reward += 10000
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.updatestate(p1_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p1.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p1.next_state_Q_value = outputs_numpy
						valid_outputs = outputs[tuple(self.available_actions)]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value

					# Calculate loss and update
					with torch.set_grad_enabled(True):
						target = target.to(device)
						current_Q_value = current_Q_value.to(device)
						loss = self.loss(current_Q_value , target)
						loss.backward()
						self.p1.optimizer.step()

				reward = 1
				while(reward == 1 and not self.isEnd):
					actions = self.available_actions
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					p2_action = self.p2.chooseAction(actions, self.board)
					current_Q_value = self.next_state_Q_value[p1_action]
					reward = self.reward(p2_action , self.p2.symbol)
					if self.winner() == p2.symbol:
						reward += 10000
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.updatestate(p1_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p2.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p2.next_state_Q_value = outputs_numpy
						valid_outputs = outputs[tuple(self.available_actions)]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value
					with torch.set_grad_enable(True):
						target = target.to(device)
						current_Q_value = current_Q_value.to(device)
						loss = self.loss(current_Q_value , target)
						loss.backward()
						self.p2.optimizer.step()
			self.p1.savePolicy()
			self.p2.savePolicy()
			self.p1.scheduler.step()
			self.p2.scheduler.step()



						
				# board_hash = self.getHash()
				# self.p1.addState(board_hash)
				# check board status if it is end

		 
	# def available_positions


class Player:
	def __init__(self, name, size, symbol, exp_rate=0.3):
		self.name = name
		# self.states = []  # record all positions taken
		# self.lr = 0.2
		# self.exp_rate = exp_rate
		# self.decay_gamma = 0.9
		self.model = Model(size).to(device)
		self.current_Q_value = 0
		self.next_state_Q_value = []
		self.symbol = symbol
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
		self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)
	# def getHash(self, board):
	# 	boardHash = str(board.reshape(board.shape[0] * bpard.shape[1] * board.shape[2]))
	# 	return boardHash

	def chooseAction(self, actions, current_board):
		if np.random.uniform(0, 1) <= self.exp_rate:
			# take random action
			# idx = np.random.choice(len(positions))
			action = np.random.choice(actions)
			self.random_action_taken = True
		else:
			outputs = self.next_state_Q_value
			valid_outputs = outputs[tuple(actions)]
			action = actions[np.argmax(valid_outputs)]
			self.current_Q_value = np.max(valid_outputs)
			self.random_action_taken = False
		# print("{} takes action {}".format(self.name, action))
		return action

	# append a hash state
	# def addState(self, state):
	# 	self.states.append(state)

	# # at the end of game, backpropagate and update states value
	# def feedReward(self, reward):
	# 	for st in reversed(self.states):
	# 		if self.states_value.get(st) is None:
	# 			self.states_value[st] = 0
	# 		self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
	# 		reward = self.states_value[st]

	# def reset(self):
	# 	self.states = []

	def savePolicy(self):
		torch.save(self.model.state_dict() , name + "_Q_dict")

	# def loadPolicy(self, file):
	# 	fr = open(file, 'rb')
	# 	self.states_value = pickle.load(fr)
	# 	fr.close()



if __name__ == "__main__":
	# training
	size = int(sys.argv[1])
	p1 = Player("p1" , size , 1)
	p2 = Player("p2" , size , -1)

	st = State(p1, p2 , size)
	print("training...")
	st.play(50000)

	# play with human
	# p1 = Player("computer", exp_rate=0)
	# p1.loadPolicy("policy_p1")

	# p2 = HumanPlayer("human")

	# st = State(p1, p2)
	# st.play2()
