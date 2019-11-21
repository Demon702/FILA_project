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
import copy
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class State:
	def __init__(self, p1, p2 , size = 10):
		self.size = size
		self.total_squares = (size - 1)**2
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		self.old_board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		self.p1 = p1
		self.p2 = p2
		self.isEnd = False
		self.boardHash = None
		# init p1 plays first
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)
		self.loss = torch.nn.MSELoss()
		self.winners = []

	def getHash(self):
		string = str(self.board.reshape((size - 1) * size * 2))
		self.boardHash = int(string , 2)
		return self.boardHash

	def reset(self):
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.isEnd = False
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)

	def available_actions(self):
		available_array = np.where(self.board == 0)
		available_array = np.array(np.transpose(available_array))
		available_actions = [2*(self.size - 1)*index[0] + 2*index[1] + index[2] for index in available_array]
		return available_actions

	def updatestate(self, action):
		[x , y , z] = [action//(2*(self.size - 1)) , (action % (2*(self.size - 1)))//2 , action % 2]
		self.board[x , y, z] = 1

	def winner(self):
		# print("P1 has won :" , sum(sum(self.grid_winners == 1)))
		# print("P2 has won :" , sum(sum(self.grid_winners == -1)))
		# print(self.grid_winners.size/2)
		# print(self.no_of_boards)
		if sum(sum(self.grid_winners == 1)) > self.grid_winners.size/2:
			self.isEnd = True
			return 1
		elif sum(sum(self.grid_winners == -1)) > self.grid_winners.size/2:
			self.isEnd = True
			return -1
		else:
			return 0

	def reward(self , action , player):
		[x,  y , z] = [action//(2*(self.size - 1)) , (action % (2*(self.size - 1)))//2 , action % 2]
		reward = 0
		if z == 0:
			if x > 0 and self.board[x-1 , y, 0] and self.board[y, x-1, 1] and self.board[y + 1, x - 1, 1]:
				self.grid_winners[x-1 , y] = player
				self.no_of_boards += 1
				reward += 1

			if x < size - 1 and self.board[x+1 , y, 0] and self.board[y, x, 1] and self.board[y+1, x, 1]:
				self.grid_winners[x , y] = player
				self.no_of_boards += 1
				reward += 1
		elif z == 1:
			if x < size - 1 and self.board[y+1, x, 0] and self.board[y, x, 0] and self.board[x+1, y, 1]:
				self.grid_winners[y , x] = player
				self.no_of_boards += 1
				reward += 1
			if x > 0  and self.board[x-1 , y, 1] and self.board[y, x-1, 0] and self.board[y+1, x-1, 0]:
				self.grid_winners[y , x-1] = player
				self.no_of_boards += 1
				reward += 1
		return reward

	def play(self, rounds=100):
		for i in range(rounds):
			if i % 1000 == 0:
				print("Rounds {}".format(i))
			while not self.isEnd:
				# Player 1
				reward = 1
				with torch.no_grad():
					tr = transforms.ToTensor()
					outputs = self.p1.model(tr(self.board))
				outputs_numpy = outputs.numpy()

				self.p1.next_state_Q_value = outputs_numpy

				while reward == 1 and not self.isEnd:
					# print("---------------------------------------------------------")
					actions = self.available_actions()
					# print(actions)
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					p1_action = self.p1.chooseAction(actions, self.board)
					current_Q_value = self.p1.next_state_Q_value[p1_action]
					reward = self.reward(p1_action , self.p1.symbol)
					winner = self.winner()
					self.winners.append(winner)
					if winner == p1.symbol:
						reward += 10000
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.old_board = self.board.copy()
						self.updatestate(p1_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p1.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p1.next_state_Q_value = outputs_numpy
						# print(outputs_numpy)
						# print(self.grid_winners)
						# print(self.available_actions())
						valid_outputs = outputs_numpy[self.available_actions()]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value
					# if p1_action in [16, 34, 145, 163]:
					# print( reward , "p1")

					# Calculate loss and update
					with torch.set_grad_enabled(True):
						target = torch.tensor(target).cpu().float()
						tr = transforms.ToTensor()
						current_Q_value = self.p1.model(tr(self.old_board))	[p1_action]
						loss = self.loss(current_Q_value , target)
						loss.backward()
						self.p1.optimizer.step()

				reward = 1
				with torch.no_grad():
					tr = transforms.ToTensor()
					outputs = self.p2.model(tr(self.board))
				outputs_numpy = outputs.numpy()
				self.p2.next_state_Q_value = outputs_numpy
				while(reward == 1 and not self.isEnd):
					# print("-------------------------------------------------------")
					actions = self.available_actions()
					# print(len(actions))
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					p2_action = self.p2.chooseAction(actions, self.board)
					current_Q_value = self.p2.next_state_Q_value[p2_action]
					reward = self.reward(p2_action , self.p2.symbol)
					winner = self.winner()
					self.winners.append(winner)
					if winner == p2.symbol:
						reward += 10000
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.old_board = self.board.copy()
						self.updatestate(p2_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p2.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p2.next_state_Q_value = outputs_numpy
						# print(len(self.available_actions))
						# print(self.grid_winners)
						# print(self.available_actions())
						valid_outputs = outputs_numpy[self.available_actions()]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value
					# if p2_action in [16, 34, 145, 163]:
					# print(p2_action , reward , "p2")
					with torch.set_grad_enabled(True):
						target = torch.tensor(target).cpu().float()
						tr = transforms.ToTensor()
						current_Q_value = self.p2.model(tr(self.old_board))	[p2_action]
						loss = self.loss(current_Q_value , target)
						loss.backward()
						self.p2.optimizer.step()
			self.p1.savePolicy()
			self.p2.savePolicy()
			self.p1.scheduler.step()
			self.p2.scheduler.step()
			# print(self.board)
			# print(self.grid_winners)
			self.reset()


						
				# board_hash = self.getHash()
				# self.p1.addState(board_hash)
				# check board status if it is end

		 
	# def available_positions


class Player:
	def __init__(self, name, size, symbol, exp_rate=0.3):
		self.name = name
		self.exp_rate = exp_rate
		# self.states = []  # record all positions taken
		# self.lr = 0.2
		# self.exp_rate = exp_rate
		# self.decay_gamma = 0.9
		self.model = Model(size).cpu()
		self.current_Q_value = 0
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
			# print(actions)
			action = np.random.choice(actions)
			self.random_action_taken = True
		else:
			outputs = self.next_state_Q_value
			# outputs = outputs.numpy()
			# print(outputs.shape)
			# print(actions)
			valid_outputs = outputs[actions]
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
		torch.save(self.model.state_dict() , 'data/' + self.name + "_Q_dict")

	# def loadPolicy(self, file):
	# 	fr = open(file, 'rb')
	# 	self.states_value = pickle.load(fr)
	# 	fr.close()


class Screen:
	def __init__(self, n=3):
		# initialize pygame
		pygame.init()
		self.gap = 40
		self.n = n
		# set the display size (one slot has 30x30 pixels; Walls: 4x26 Box: 26x26)
		self.screen = pygame.display.set_mode([n*30 + self.gap + 4, 30*n + self.gap + 4])

		# load all images
		self.empty = pygame.image.load("pics/empty.png")
		self.A = pygame.image.load("pics/A.png")
		self.B = pygame.image.load("pics/B.png")
		self.X = pygame.image.load("pics/X.png")
		self.block = pygame.image.load("pics/block.png")
		self.lineX = pygame.image.load("pics/lineX.png")
		self.lineXempty = pygame.image.load("pics/lineXempty.png")
		self.lineY = pygame.image.load("pics/lineY.png")
		self.lineYempty = pygame.image.load("pics/lineYempty.png")
		self.color = [155, 155, 155]
		self.screen.fill(self.color)
		pygame.display.update()
		# self.show()



	def show(self, p, wins):
		print("P : \n", p)
		print("wins : \n:", wins)
		for i in range(self.n):
			for j in range(self.n):
				x, y = i*30 + self.gap, j*30 + self.gap
				self.screen.blit(self.block, (x, y))
		for r in range(len(p)):
			for c in range(len(p[r])):

				k1 = p[r][c][0]
				x, y = c*30 + self.gap, r*30 + self.gap
				x += 4
				# y += 4
				if not k1:
					self.screen.blit(self.lineXempty, (x, y))
				else:
					self.screen.blit(self.lineX, (x, y))

				k2 = p[r][c][1]
				x, y = r*30 + self.gap, c*30 + self.gap
				y += 4
				if not k2:
					self.screen.blit(self.lineYempty, (x, y))
				else:
					self.screen.blit(self.lineY, (x, y))

		for r in range(len(wins)):
			for c in range(len(wins[r])):
				x, y = c*30 + self.gap, r*30 + self.gap
				x += 4
				y += 4
				k = wins[r][c]
				if k == 0:
					self.screen.blit(self.empty, (x, y))
				elif k == 1:
					self.screen.blit(self.A, (x, y))
				elif k == -1:
					self.screen.blit(self.B, (x, y))


		pygame.display.flip()



if __name__ == "__main__":
	# training
	# size = int(sys.argv[1])
	# p1 = Player("p1" , size , 1)
	# p2 = Player("p2" , size , -1)

	# st = State(p1, p2 , size)
	# print("training...")
	# st.play(10000)
	# st.winners = np.array(st.winners)
	# print(sum(st.winners == 1))
	# print(sum(st.winners == -1))


	# play with human
	# p1 = Player("computer", exp_rate=0)
	# p1.loadPolicy("policy_p1")

	# p2 = HumanPlayer("human")

	# st = State(p1, p2)
	# st.play2()
	n = int(sys.argv[1])
	scr = Screen(n)
	p = np.random.randint(2, size=(n, n-1, 2))
	wins = np.random.randint(3, size=(n-1, n-1))
	wins -= 1
	# p = np.load('p.npy')
	# while True:
	import time
	scr.show(p, wins)
	time.sleep(30)

